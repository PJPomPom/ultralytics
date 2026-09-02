import time
import yaml
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from ultralytics.data.augment import LetterBox
from ultralytics.utils.metrics import ap_per_class, ConfusionMatrix

# ============================================================
# 配置
# ============================================================
MODELS = [
    {"name": "yolo26s.onnx", "end2end": False, "desc": "YOLO26s 标准版"},
    {"name": "yolo26s_end2end.onnx", "end2end": True, "desc": "YOLO26s 端到端版"},
    {"name": "yolov8s.onnx", "end2end": False, "desc": "YOLOv8s 标准版"},
]
DATASET_YAML = "ultralytics/cfg/datasets/coco.yaml"
CONF_THRES = 0.001  # 评估时用低阈值，收集所有预测用于计算 mAP
IOU_THRES = 0.6
IMG_SIZE = 640
# cpu: "CPUExecutionProvider"  |  gpu: "CUDAExecutionProvider"
DEVICES = "CUDAExecutionProvider"

# ============================================================
# 路径
# ============================================================
script_dir = Path(__file__).resolve().parent
proj_dir = script_dir.parent
img_dir = script_dir / "images" / "val"
label_dir = script_dir / "labels" / "val"

with open(proj_dir / DATASET_YAML, encoding="utf-8") as f:
    names_dict = yaml.safe_load(f)["names"]
    CLASS_NAMES = [names_dict[i] for i in sorted(names_dict)]
NUM_CLASSES = len(CLASS_NAMES)

# 设备名称
device_name = DEVICES.replace("ExecutionProvider", "")

# 收集所有图片路径
img_paths = sorted(img_dir.glob("*.jpg"))
if not img_paths:
    img_paths = sorted(img_dir.glob("*.png"))


# ============================================================
# NMS
# ============================================================
def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # 从高到低排序
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=np.int64)


# ============================================================
# 单张图片推理
# ============================================================
def predict_one(session, input_name, img0, is_end2end, transform):
    """推理单张图片，返回 (boxes_xyxy_abs, scores, class_ids) 和耗时"""
    h0, w0 = img0.shape[:2]
    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    im = transform(image=img_rgb)
    im = im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    t1 = time.perf_counter()

    output = session.run(None, {input_name: im})[0]
    t2 = time.perf_counter()

    if is_end2end:
        pred = output[0]  # (300, 6) [x1, y1, x2, y2, conf, cls]
        mask = pred[:, 4] > CONF_THRES
        boxes, scores, class_ids = pred[mask, :4], pred[mask, 4], pred[mask, 5].astype(int)
    else:
        output = output.squeeze(0).transpose(1, 0)  # (8400, 84)
        boxes = output[:, :4]
        scores_all = output[:, 4:]
        boxes = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2,
                                boxes[:, :2] + boxes[:, 2:] / 2], axis=1)
        max_scores = scores_all.max(axis=1)
        class_ids = scores_all.argmax(axis=1)
        mask = max_scores > CONF_THRES
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]
        keep = nms(boxes, max_scores, IOU_THRES)
        boxes, scores, class_ids = boxes[keep], max_scores[keep], class_ids[keep]

    t3 = time.perf_counter()

    # 坐标缩放回原图
    if len(boxes) > 0:
        gain = min(IMG_SIZE / h0, IMG_SIZE / w0)
        pad_w = (IMG_SIZE - w0 * gain) / 2
        pad_h = (IMG_SIZE - h0 * gain) / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain
        boxes = boxes.clip(0, max(w0, h0))

    return boxes, scores, class_ids, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)


# ============================================================
# 加载标签
# ============================================================
def load_labels(label_path, img_w, img_h):
    """加载 YOLO 格式标签，返回 (boxes_xyxy_abs, class_ids)"""
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    gt = np.loadtxt(label_path, ndmin=2)
    if len(gt) == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    cls_ids = gt[:, 0].astype(int)
    # 归一化 xywh → 绝对 xyxy
    cx, cy, w, h = gt[:, 1] * img_w, gt[:, 2] * img_h, gt[:, 3] * img_w, gt[:, 4] * img_h
    boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return boxes, cls_ids


# ============================================================
# 计算 IoU
# ============================================================
def box_iou(box1, box2):
    """计算两组框的 IoU 矩阵，box1 (N,4), box2 (M,4) → (N, M)"""
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    iou = inter / (area1[:, None] + area2[None, :] - inter + 1e-16)
    return iou


# ============================================================
# 逐个模型评估
# ============================================================
transform = LetterBox(new_shape=IMG_SIZE, auto=False, stride=32)

for cfg in MODELS:
    onnx_path = script_dir / cfg["name"]
    if not onnx_path.exists():
        print(f"\n{'=' * 60}")
        print(f"跳过: {cfg['name']} 不存在")
        continue

    print(f"\n{'=' * 60}")
    print(f"模型: {cfg['name']}  ({cfg['desc']})")
    print(f"推理设备: {device_name}  |  图片数: {len(img_paths)}")

    session = ort.InferenceSession(str(onnx_path), providers=[DEVICES])
    input_name = session.get_inputs()[0].name

    # 预热
    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    for _ in range(5):
        session.run(None, {input_name: dummy})

    # 收集所有预测和标签
    all_preds = []  # 每个元素: [x1, y1, x2, y2, conf, cls]
    all_gts = []  # 每个元素: [x1, y1, x2, y2, cls]
    times_pre, times_infer, times_post, times_total = [], [], [], []

    for img_path in img_paths:
        img_name = img_path.stem
        label_path = label_dir / f"{img_name}.txt"

        img0 = cv2.imread(str(img_path))
        h0, w0 = img0.shape[:2]

        boxes, scores, class_ids, t = predict_one(session, input_name, img0, cfg["end2end"], transform)
        times_pre.append(t[0])
        times_infer.append(t[1])
        times_post.append(t[2])
        times_total.append(t[3])

        # 收集预测结果
        if len(boxes) > 0:
            for b, s, c in zip(boxes, scores, class_ids):
                all_preds.append([*b, s, int(c)])

        # 收集标签
        gt_boxes, gt_cls = load_labels(label_path, w0, h0)
        if len(gt_boxes) > 0:
            for b, c in zip(gt_boxes, gt_cls):
                all_gts.append([*b, int(c)])

    all_preds = np.array(all_preds) if all_preds else np.zeros((0, 6))
    all_gts = np.array(all_gts) if all_gts else np.zeros((0, 5))

    # ============================================================
    # 打印耗时
    # ============================================================
    print(f"\n{'':>8s} {'均值 (ms)':>10s} {'最小 (ms)':>10s} {'最大 (ms)':>10s}")
    print(
        f"{'预处理':>8s} {np.mean(times_pre) * 1000:10.3f} {np.min(times_pre) * 1000:10.3f} {np.max(times_pre) * 1000:10.3f}")
    print(
        f"{'推理':>8s} {np.mean(times_infer) * 1000:10.3f} {np.min(times_infer) * 1000:10.3f} {np.max(times_infer) * 1000:10.3f}")
    print(
        f"{'后处理':>8s} {np.mean(times_post) * 1000:10.3f} {np.min(times_post) * 1000:10.3f} {np.max(times_post) * 1000:10.3f}")
    print(
        f"{'总计':>8s} {np.mean(times_total) * 1000:10.3f} {np.min(times_total) * 1000:10.3f} {np.max(times_total) * 1000:10.3f}")

    # ============================================================
    # 计算精度指标 (mAP@0.5 和 mAP@0.5:0.95)
    # ============================================================
    if len(all_preds) == 0:
        print("\n无预测结果，跳过精度计算")
        continue

    num_gts = len(all_gts)
    num_preds = len(all_preds)
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.5, 0.55, ..., 0.95]

    # 按置信度排序
    order = np.argsort(-all_preds[:, 4])
    preds_sorted = all_preds[order]

    def compute_ap_at_iou(preds, gts, iou_thr):
        """计算给定 IoU 阈值下的 AP"""
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_matched = {}

        for i, pred in enumerate(preds):
            pred_box = pred[:4]
            pred_cls = int(pred[5])

            gt_mask = gts[:, 4] == pred_cls
            gt_candidates = gts[gt_mask]
            gt_indices = np.where(gt_mask)[0]

            if len(gt_candidates) == 0:
                fp[i] = 1
                continue

            ious = box_iou(pred_box[None], gt_candidates)[0]
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]
            best_gt_idx = gt_indices[best_iou_idx]

            if best_iou >= iou_thr:
                if pred_cls not in gt_matched:
                    gt_matched[pred_cls] = set()
                if best_gt_idx not in gt_matched[pred_cls]:
                    tp[i] = 1
                    gt_matched[pred_cls].add(best_gt_idx)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / max(num_gts, 1)
        precision = np.divide(tp_cumsum, (tp_cumsum + fp_cumsum),
                              out=np.zeros_like(tp_cumsum, dtype=float),
                              where=(tp_cumsum + fp_cumsum) > 0)

        # 11-point interpolated AP
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.any(recall >= t):
                ap += np.max(precision[recall >= t])
        ap /= 11
        return ap

    # 计算 mAP@0.5:0.95 (10 个 IoU 阈值的 AP 均值)
    ap_per_iou = []
    for iou_thr in iou_thresholds:
        ap = compute_ap_at_iou(preds_sorted, all_gts, iou_thr)
        ap_per_iou.append(ap)

    ap50 = ap_per_iou[0]  # IoU=0.5 即 mAP@0.5
    map50_95 = np.mean(ap_per_iou)  # mAP@0.5:0.95

    # 打印精度
    print(f"\n{'=' * 40}")
    print(f"精度指标 ({device_name}推理)")
    print(f"  GT 总数: {num_gts}  |  预测总数: {num_preds}")
    print(f"  mAP@0.5:      {ap50:.4f}")
    print(f"  mAP@0.5:0.95: {map50_95:.4f}")
    print(f"  各 IoU 阈值 AP:")
    for iou_thr, ap in zip(iou_thresholds, ap_per_iou):
        print(f"    IoU={iou_thr:.2f}: {ap:.4f}")

    # 打印推理设备总览
    # print(f"\n  推理设备: {device_name} ({DEVICES})")

print(f"\n{'=' * 60}")
print("评估完成")
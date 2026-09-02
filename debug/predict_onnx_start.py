import yaml
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from datetime import datetime
from ultralytics.data.augment import LetterBox
from debug.image_splitter import ImageSplitter

# ============================================================
# 配置
# ============================================================
ONNX_MODEL = "../weights/yolo26s.onnx"  # 权重在项目根目录 weights/ 下
IMG_DIR = "debug/images/predict"  # 图片文件夹路径，自动检测所有图片
DATASET_YAML = "ultralytics/cfg/datasets/coco.yaml"  # 标签配置文件

# 模型基本设置
MODEL_SIZE = 640  # 模型输入尺寸
CONF_THRES = 0.6
IOU_THRES = 0.5

# 切片检测
IS_SPLIT = True   # True False
IS_WBF = True    # True False
SPLIT_TILE_SIZE = 720   # 分割子图尺寸
SPLIT_OVERLAP = 0.2      # 分割重叠比例

# 端到端检测（仅yolo26支持）
IS_END2END = False  # 与导出时的 end2end 保持一致: True→xyxy, False→xywh

# ============================================================
# 路径
# ============================================================
script_dir = Path(__file__).resolve().parent
proj_dir = script_dir.parent
onnx_path = proj_dir / "weights" / Path(ONNX_MODEL).name  # 权重在项目根目录 weights/ 下
img_dir = Path(IMG_DIR)
if not img_dir.is_absolute():
    img_dir = (proj_dir / IMG_DIR).resolve()

# 自动扫描图片文件
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
assert len(img_files) > 0, f"文件夹 {img_dir} 中没有找到图片文件"
print(f"找到 {len(img_files)} 张图片: {[f.name for f in img_files]}")

# 加载标签名 (从 coco.yaml 的 names 字段)
with open(proj_dir / DATASET_YAML, encoding="utf-8") as f:
    names_dict = yaml.safe_load(f)["names"]
    CLASS_NAMES = [names_dict[i] for i in sorted(names_dict)]

# 模型名称 (用于打印和输出文件名)
model_name = onnx_path.stem.lower()

# 输出目录 (按模型名+功能标记+时间戳创建子文件夹)
suffix = ""
if IS_SPLIT:
    suffix += "_split"
if IS_WBF:
    suffix += "_wbf"
if IS_END2END:
    suffix += "_end2end"
out_dir = script_dir / "output" / f"{model_name}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out_dir.mkdir(parents=True, exist_ok=True)

# 创建 ONNX session (只需创建一次)
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def nms(boxes, scores, iou_thres):
    """NMS 非极大值抑制"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
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
# 循环推理所有图片
# ============================================================
for img_path in img_files:
    print(f"\n{'=' * 60}")
    print(f"推理: {img_path.name}")
    print(f"{'=' * 60}")

    # ============================================================
    # 1. 预处理
    # ============================================================
    img = cv2.imread(str(img_path))
    assert img is not None, f"图片不存在: {img_path}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if IS_SPLIT:
        # ============================================================
        # A. 分割检测模式: 大图 → 切分子图 → 分别推理 → 合并结果
        # ============================================================
        print(f"分割检测: 子图 {SPLIT_TILE_SIZE}×{SPLIT_TILE_SIZE}, 重叠 {SPLIT_OVERLAP}")

        tiles = []
        ImageSplitter.split_with_overlap(
            img_rgb, tile_w=SPLIT_TILE_SIZE, tile_h=SPLIT_TILE_SIZE,
            overlap=SPLIT_OVERLAP, out_list=tiles
        )
        print(f"切分为 {len(tiles)} 个子图")

        all_boxes = []
        all_scores = []
        all_class_ids = []

        for tile_idx, (sub_img, (tx, ty, tw, th)) in enumerate(tiles):
            # 子图预处理: LetterBox
            transform = LetterBox(new_shape=MODEL_SIZE, auto=False, stride=32)
            im = transform(image=sub_img)
            im = im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

            # 子图推理
            output = session.run(None, {input_name: im})[0]

            # 子图后处理
            output = output.squeeze(0).transpose(1, 0)
            boxes = output[:, :4]
            scores = output[:, 4:]
            boxes = np.concatenate([
                boxes[:, :2] - boxes[:, 2:] / 2,
                boxes[:, :2] + boxes[:, 2:] / 2,
            ], axis=1)
            max_scores = scores.max(axis=1)
            class_ids = scores.argmax(axis=1)

            # 置信度过滤
            mask = max_scores > CONF_THRES
            boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

            if len(boxes) > 0:
                # LetterBox 逆变换: 模型尺寸 → 子图坐标
                gain = min(MODEL_SIZE / th, MODEL_SIZE / tw)
                pad_w = (MODEL_SIZE - tw * gain) / 2
                pad_h = (MODEL_SIZE - th * gain) / 2
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain
                # 子图坐标 → 原图坐标 (加上子图在原图中的偏移)
                boxes[:, [0, 2]] += tx
                boxes[:, [1, 3]] += ty

            all_boxes.append(boxes)
            all_scores.append(max_scores)
            all_class_ids.append(class_ids)

            print(f"  子图 [{tile_idx}] ({tx},{ty} {tw}×{th}): {len(boxes)} 个检测")

        # 合并所有子图的检测结果
        if all_boxes:
            boxes = np.concatenate(all_boxes)
            max_scores = np.concatenate(all_scores)
            class_ids = np.concatenate(all_class_ids)
        else:
            boxes = np.array([]).reshape(0, 4)
            max_scores = np.array([])
            class_ids = np.array([])

        print(f"\n合并后共 {len(boxes)} 个检测框 ({'WBF' if IS_WBF else 'NMS'}前)")

        if IS_WBF:
            # WBF 加权框融合 (重叠区域的检测框按置信度加权融合，优于NMS丢弃)
            if len(boxes) > 0:
                boxes, max_scores, class_ids = ImageSplitter.wbf(
                    boxes, max_scores, class_ids, iou_thr=IOU_THRES
                )
        else:
            # NMS 去重 (重叠区域的重复检测)
            if len(boxes) > 0:
                keep = nms(boxes, max_scores, IOU_THRES)
                boxes, max_scores, class_ids = boxes[keep], max_scores[keep], class_ids[keep]

        # Clip 到原图范围
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img.shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img.shape[0])

        print(f"{'WBF' if IS_WBF else 'NMS'}后: {len(boxes)} 个检测框")

    else:
        # ============================================================
        # B. 普通模式 (原逻辑不变)
        # ============================================================
        # 缩放图片，有padding填充
        transform = LetterBox(new_shape=MODEL_SIZE, auto=False, stride=32)
        # im: (MODEL_SIZE, MODEL_SIZE, 3)
        im = transform(image=img_rgb)
        # [None] 等价于 np.newaxis，在最前面插入一个轴，转换为浮点，然后归一化到 [0, 1]
        im = im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        # ============================================================
        # 2. ONNX 推理
        # ============================================================
        output = session.run(None, {input_name: im})[0]  # (1, 84, 8400)

        # ============================================================
        # 3. 后处理
        # ============================================================
        if IS_END2END:
            # 端到端: (1, 300, 6) → 每行 [x1, y1, x2, y2, conf, class_id]
            pred = output[0]  # (300, 6)
            boxes = pred[:, :4]  # (300, 4) xyxy 格式
            max_scores = pred[:, 4]  # (300,) 置信度
            class_ids = pred[:, 5].astype(int)  # (300,) 类别ID
        else:
            # 标准版: (1, 84, 8400) → (8400, 84)
            output = output.squeeze(0).transpose(1, 0)  # (8400, 84)
            boxes = output[:, :4]  # (8400, 4) xywh 格式
            scores = output[:, 4:]  # (8400, 80) 已 sigmoid

            # xywh → xyxy
            boxes = np.concatenate([
                boxes[:, :2] - boxes[:, 2:] / 2,
                boxes[:, :2] + boxes[:, 2:] / 2,
            ], axis=1)

            max_scores = scores.max(axis=1)
            class_ids = scores.argmax(axis=1)

        # 置信度过滤
        mask = max_scores > CONF_THRES
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

        # ============================================================
        # 阶段1: 打印网络直接输出的检测框 (置信度过滤后, NMS前, 模型输入尺寸空间)
        # ============================================================
        print("=" * 60)
        print(f"模型: {ONNX_MODEL}  |  类型: {'end2end (xyxy)' if IS_END2END else '标准 (xywh→xyxy)'}")
        print(f"ONNX 输出形状: {output.shape}")
        print(f"\n--- 阶段1: 网络直接输出 (置信度过滤后, NMS前, {MODEL_SIZE}×{MODEL_SIZE}空间) ---")
        print(f"检测数: {len(boxes)} (仅显示前 10 个)")
        print(f"{'x1':>7s} {'y1':>7s} {'x2':>7s} {'y2':>7s}  {'conf':>6s}  class")
        show_n = min(10, len(boxes))
        for i in range(show_n):
            print(f"{boxes[i][0]:7.1f} {boxes[i][1]:7.1f} {boxes[i][2]:7.1f} {boxes[i][3]:7.1f}  "
                  f"{max_scores[i]:.4f}  {CLASS_NAMES[int(class_ids[i])]}")
        if len(boxes) > 10:
            print(f"  ... 省略 {len(boxes) - 10} 个")

        # NMS (端到端版只需置信度过滤，标准版需 IoU 去重)
        if not IS_END2END and len(boxes) > 0:
            keep = nms(boxes, max_scores, IOU_THRES)
            boxes, max_scores, class_ids = boxes[keep], max_scores[keep], class_ids[keep]

        # 坐标缩放回原图 (LetterBox 逆变换)
        gain = min(MODEL_SIZE / img.shape[0], MODEL_SIZE / img.shape[1])
        pad_w = (MODEL_SIZE - img.shape[1] * gain) / 2
        pad_h = (MODEL_SIZE - img.shape[0] * gain) / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain
        boxes = boxes.clip(0, max(img.shape[1], img.shape[0]))

    # ============================================================
    # 阶段2: 打印后处理后的最终检测框 (WBF/NMS后, 已映射回原图坐标)
    # ============================================================
    print(f"\n--- 阶段2: {'WBF' if IS_WBF else 'NMS'}/后处理 最终结果 (已映射回原图坐标) ---")
    print(f"检测数: {len(boxes)}")
    print(f"{'x1':>7s} {'y1':>7s} {'x2':>7s} {'y2':>7s}  {'conf':>6s}  class")
    for i in range(len(boxes)):
        print(f"{boxes[i][0]:7.1f} {boxes[i][1]:7.1f} {boxes[i][2]:7.1f} {boxes[i][3]:7.1f}  "
              f"{max_scores[i]:.4f}  {CLASS_NAMES[int(class_ids[i])]}")

    # ============================================================
    # 5. 画框并保存
    # ============================================================
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
              (0, 255, 255), (255, 0, 255), (128, 0, 255), (255, 128, 0)]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        color = colors[int(class_ids[i]) % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[int(class_ids[i])]} {max_scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)

    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"推理结果已保存: {out_path}")
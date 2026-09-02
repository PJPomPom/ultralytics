import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# ============================================================
# 配置
# ============================================================
MODE = "text"  # "text" = 文本提示, "pf" = prompt-free, "visual" = 视觉提示
MODEL_NAME = "yoloe-26s-seg.pt"  # 预训练权重，首次运行会自动下载
IMG_NAME = "street001.jpg"  # 推理图片名
CONF_THRES = 0.4  # 置信度阈值
# "person", "bus", "van", "car", "city square", "traffic sign", "helmet", "traffic light", "speed limit"
TEXT_CLASSES = ["light",
                # "bus",
                # "bicycle",
                # "car",
                # "motorcycle"
                ]  # 文本提示的类别

# 视觉提示配置 (仅 MODE="visual" 时使用)
# 给模型一个参考框，让它找类似目标
VISUAL_PROMPTS = {
    "bboxes": np.array([[221.52, 405.8, 344.98, 857.54]]),  # 参考框 xyxy
    "cls": np.array([0]),  # 类别索引
}

# ============================================================
# 路径
# ============================================================
script_dir = Path(__file__).resolve().parent
proj_dir = script_dir.parent
img_path = proj_dir / "ultralytics" / "assets" / IMG_NAME
out_dir = script_dir / "output"
out_dir.mkdir(exist_ok=True)

assert img_path.exists(), f"图片不存在: {img_path}"


# ============================================================
# 打印结果
# ============================================================
def print_results(results, model):
    print("=" * 60)
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print("未检测到任何目标")
            continue
        print(f"检测到 {len(boxes)} 个目标:")
        print(f"{'x1':>7s} {'y1':>7s} {'x2':>7s} {'y2':>7s}  {'conf':>6s}  class")
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            class_name = model.names[cls[i]] if isinstance(model.names, dict) else model.names[cls[i]]
            print(f"{xyxy[i][0]:7.1f} {xyxy[i][1]:7.1f} {xyxy[i][2]:7.1f} {xyxy[i][3]:7.1f}  "
                  f"{conf[i]:.4f}  {class_name}")
    print("=" * 60)


# ============================================================
# 主流程
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if MODE == "text":
    # ============================================================
    # 方式1: 文本提示模式 (Text Prompt)
    # 用文字描述要检测的类别，模型通过 mobileclip2 文本编码器将类别名转为嵌入向量
    # 优点: 灵活，可以检测任意类别
    # 缺点: 首次 set_classes 需要加载文本编码器
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"YOLOE 文本提示模式 (Text Prompt)")
    print(f"模型: {MODEL_NAME}")
    print(f"类别: {TEXT_CLASSES}")
    print(f"{'=' * 60}")

    model = YOLO(MODEL_NAME)
    model.set_classes(TEXT_CLASSES)

    results = model.predict(
        source=str(img_path),
        conf=CONF_THRES,
        save=True,
        project=str(out_dir),
        name=f"yoloe_text_{timestamp}",
    )

    print_results(results, model)

elif MODE == "pf":
    # ============================================================
    # 方式2: Prompt-Free 模式 (固定类别)
    # 将文本嵌入融合进模型权重，之后推理不需要文本编码器，速度更快
    # 优点: 纯卷积推理，速度快
    # 缺点: 类别固定，不可更改
    #
    # 两种情况:
    #   a) 使用普通模型 (如 yoloe-26s-seg.pt) → 需要 set_vocab 融合类别
    #   b) 使用预融合模型 (如 yoloe-26s-seg-pf.pt) → 直接推理，无需融合
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"YOLOE Prompt-Free 模式 (固定类别)")
    print(f"模型: {MODEL_NAME}")
    print(f"类别: {TEXT_CLASSES}")
    print(f"{'=' * 60}")

    model = YOLOE(MODEL_NAME)

    # 判断是否是已融合的 prompt-free 模型
    is_already_pf = "-pf" in MODEL_NAME or hasattr(model.model.model[-1], "lrpc")
    if is_already_pf:
        print("模型已是 prompt-free 格式，直接推理")
        print(f"模型实际类别数: {model.model.model[-1].nc}")
        names = model.model.names
        if isinstance(names, dict):
            print(f"模型实际类别: {list(names.values())[:10]}...")
    else:
        print("正在将文本嵌入融合到模型权重...")
        model.set_vocab(model.get_vocab(TEXT_CLASSES), TEXT_CLASSES)
        print("融合完成!")

    # 先用低阈值试一下，看模型是否正常输出
    results = model.predict(
        source=str(img_path),
        conf=CONF_THRES,
        save=True,
        project=str(out_dir),
        name=f"yoloe_pf_{timestamp}",
    )

    print_results(results, model)

elif MODE == "visual":
    # ============================================================
    # 方式3: 视觉提示模式 (Visual Prompt)
    # 给模型一个参考框，让它检测图像中相似的目标
    # 优点: 不需要文字描述，用示例框来指定目标
    # 缺点: 需要提供参考框
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"YOLOE 视觉提示模式 (Visual Prompt)")
    print(f"模型: {MODEL_NAME}")
    print(f"参考框: {VISUAL_PROMPTS['bboxes']}")
    print(f"{'=' * 60}")

    model = YOLO(MODEL_NAME)

    results = model.predict(
        source=str(img_path),
        visual_prompts=VISUAL_PROMPTS,
        predictor=YOLOEVPSegPredictor,
        conf=CONF_THRES,
        save=True,
        project=str(out_dir),
        name=f"yoloe_visual_{timestamp}",
    )

    print_results(results, model)

else:
    raise ValueError(f"不支持的 MODE: {MODE}，可选: 'text' / 'pf' / 'visual'")

print(f"\n推理完成! 结果已保存到: {out_dir}")

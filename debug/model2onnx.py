from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "yolo26n_comps_10.pt"  # 改为 "yolov8s.pt" 即可切换模型
END2END = False  # True: 输出 (1,300,6) 含后处理; False: 输出 (1,84,8400) 原始
SIMPLIFY = True  # 是否 onnxslim 简化 ONNX 图结构，默认开启
OPSET = 12  # ONNX opset 版本
HALF = False  # 是否导出 FP16

# ============================================================
# 路径
# ============================================================
script_dir = Path(__file__).resolve().parent
model_path = script_dir / MODEL_NAME

# ============================================================
# 导出
# ============================================================
model = YOLO(str(model_path))

model.export(
    format="onnx",
    end2end=END2END,
    simplify=SIMPLIFY,
    opset=OPSET,
    half=HALF,
)



print(f"\nONNX 模型已导出: {script_dir / model_path.stem}.onnx")

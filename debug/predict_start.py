import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

# 1. 加载模型
model = YOLO("../weights/yolo26s.pt", verbose=False)

# 2. 预处理 - 使用脚本所在目录为基准的绝对路径
script_dir = Path(__file__).resolve().parent
img_path = script_dir.parent / "ultralytics" / "assets" / "bus.jpg"
img = cv2.imread(str(img_path))
assert img is not None, f"图片不存在: {img_path}"
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = LetterBox(new_shape=640, auto=False, stride=32)
im = transform(image=img)
im = torch.from_numpy(im.transpose(2, 0, 1)[None]).float() / 255.0
im = im.to(model.device)

# 3. 获取 Detect 头
detect = model.model.model[-1]  # 最后一个模块是 Detect 头

# 4. 截取 _inference 中间结果 (monkey-patch) 和 forward 最终结果 (hook)
raw_inference = None   # _inference 输出 (1, 84, 8400)
final_output = None    # forward 最终输出 (1, 300, 6)

# _inference 是普通方法，不能注册 hook，用 monkey-patch 替代
_original_inference = detect._inference

def patched_inference(self, x):
    global raw_inference
    raw_inference = _original_inference(x)
    return raw_inference

detect._inference = patched_inference.__get__(detect, type(detect))

def hook_forward(module, input, output):
    global final_output
    final_output = output[0] if isinstance(output, tuple) else output

hook2 = detect.register_forward_hook(hook_forward)

# 5. 推理
with torch.no_grad():
    model.model(im)

hook2.remove()

# 6. 打印 _inference 中间结果 (decode后、postprocess前)
print("=" * 60)
print("【_inference 中间结果 - decode后、postprocess前】")
print(f"形状: {raw_inference.shape}")  # (1, 84, 8400)
print(f"前 4 通道 (坐标 x1,y1,x2,y2): \n{raw_inference[0, :4, :10]}")
print(f"后 80 通道 (类别分数, 已 sigmoid): \n{raw_inference[0, 4:, :10]}")

# 7. 打印 forward 最终结果 (postprocess后)
print("=" * 60)
print("【forward 最终结果 - postprocess后】")
print(f"形状: {final_output.shape}")  # (1, 300, 6)
print(f"内容: [x1, y1, x2, y2, confidence, class_id]")
print(f"前 10 个检测结果:\n{final_output[0, :10, :]}")
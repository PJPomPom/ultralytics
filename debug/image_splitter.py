import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class ImageSplitter:
    @staticmethod
    def split_by_grid(img, rows=2, cols=2):
        """
        将图片按网格切分成 rows × cols 个子图

        Args:
            img: 图片路径 (str/Path) 或 numpy 数组 (H, W, C)
            rows: 行数
            cols: 列数

        Returns:
            list of (sub_img, (x, y, w, h)) 子图及其在原图中的位置
        """
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            assert img is not None, f"图片不存在: {img}"

        h, w = img.shape[:2]
        tile_h = h // rows
        tile_w = w // cols

        results = []
        for r in range(rows):
            for c in range(cols):
                y1 = r * tile_h
                x1 = c * tile_w
                y2 = y1 + tile_h if r < rows - 1 else h
                x2 = x1 + tile_w if c < cols - 1 else w
                sub = img[y1:y2, x1:x2]
                results.append((sub, (x1, y1, x2 - x1, y2 - y1)))
        return results

    @staticmethod
    def split_by_size(img, tile_w=640, tile_h=640, stride_w=0, stride_h=0, out_list=None):
        """
        按指定尺寸切分图片，支持重叠

        Args:
            img: 图片路径 (str/Path) 或 numpy 数组 (H, W, C)
            tile_w: 子图宽度
            tile_h: 子图高度
            stride_w: 水平步长，0 表示等于 tile_w (不重叠)
            stride_h: 垂直步长，0 表示等于 tile_h (不重叠)
            out_list: 外部传入的列表，如果不为 None，结果将追加到此列表中 (按从左到右、从上到下顺序)

        Returns:
            list of (sub_img, (x, y, w, h))，如果 out_list 不为 None 则返回 out_list
        """
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            assert img is not None, f"图片不存在: {img}"

        h, w = img.shape[:2]
        stride_w = stride_w or tile_w
        stride_h = stride_h or tile_h

        results = out_list if out_list is not None else []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x2 = min(x + tile_w, w)
                y2 = min(y + tile_h, h)
                sub = img[y:y2, x:x2]
                results.append((sub, (x, y, x2 - x, y2 - y)))
                x += stride_w
            y += stride_h

        return results

    @staticmethod
    def save_splits(splits, out_dir, prefix="tile", ext=".jpg"):
        """
        保存切分后的子图到文件夹

        Args:
            splits: split_by_grid / split_by_size 返回的结果列表
            out_dir: 输出目录 (str/Path)
            prefix: 文件名前缀
            ext: 文件扩展名

        Returns:
            list of Path 保存的文件路径
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for i, (sub, _) in enumerate(splits):
            path = out_dir / f"{prefix}_{i:04d}{ext}"
            cv2.imwrite(str(path), sub)
            saved.append(path)

        print(f"已保存 {len(saved)} 张子图到: {out_dir}")
        return saved

    @staticmethod
    def merge_splits(splits, original_shape, tile_w, tile_h, stride_w=0, stride_h=0):
        """
        将切分后的子图合并回原图

        Args:
            splits: list of numpy 子图数组
            original_shape: 原图尺寸 (H, W)
            tile_w: 子图宽度
            tile_h: 子图高度
            stride_w: 水平步长
            stride_h: 垂直步长

        Returns:
            numpy 数组，合并后的完整图像
        """
        h, w = original_shape[:2]
        merged = np.zeros((h, w, 3), dtype=np.uint8)
        stride_w = stride_w or tile_w
        stride_h = stride_h or tile_h

        idx = 0
        y = 0
        while y < h and idx < len(splits):
            x = 0
            while x < w and idx < len(splits):
                sub = splits[idx]
                sh, sw = sub.shape[:2]
                merged[y:y + sh, x:x + sw] = sub
                x += stride_w
                idx += 1
            y += stride_h

        return merged

    @staticmethod
    def split_with_overlap(img, tile_w=640, tile_h=640, overlap=0.2, out_list=None):
        """
        按指定尺寸 + 重叠比例切分图片

        Args:
            img: 图片路径或 numpy 数组
            tile_w: 子图宽度
            tile_h: 子图高度
            overlap: 重叠比例 (0.0 ~ 1.0)
            out_list: 外部传入的列表，如果不为 None，结果将按从左到右、从上到下顺序追加到此列表中

        Returns:
            list of (sub_img, (x, y, w, h))，如果 out_list 不为 None 则返回 out_list
        """
        stride_w = int(tile_w * (1 - overlap))
        stride_h = int(tile_h * (1 - overlap))
        return ImageSplitter.split_by_size(img, tile_w, tile_h, stride_w, stride_h, out_list)

    @staticmethod
    def map_boxes_to_original(boxes, sub_img_pos, model_size=640):
        """
        将模型输出空间 (model_size × model_size) 中的检测框映射回原图坐标

        坐标变换链:
            原图 (5472×3648) → 子图 (1024×1024) → 模型输入 (640×640)
            模型输出 (640×640) → 子图 (1024×1024) → 原图 (5472×3648)

        Args:
            boxes: 检测框列表，每个框格式为 [x1, y1, x2, y2, ...] 在 model_size 空间中
            sub_img_pos: (x, y, w, h) 子图在原图中的位置，由 split_by_size/split_by_grid 返回
            model_size: 模型输入尺寸，默认 640

        Returns:
            list of boxes（格式同输入），坐标已映射到原图空间
        """
        sx, sy, sw, sh = sub_img_pos
        scale_x = sw / model_size
        scale_y = sh / model_size

        mapped = []
        for box in boxes:
            b = box.copy()
            b[0] = box[0] * scale_x + sx
            b[1] = box[1] * scale_y + sy
            b[2] = box[2] * scale_x + sx
            b[3] = box[3] * scale_y + sy
            mapped.append(b)

        return mapped

    @staticmethod
    def wbf(boxes, scores, class_ids, iou_thr=0.6):
        """
        Weighted Boxes Fusion (WBF) - 加权框融合

        与 NMS 不同，WBF 不会丢弃重叠的框，而是将属于同一目标的多个框
        按置信度加权融合为一个更精确的框。特别适合分割检测场景中，
        目标跨子图边界时各子图检测到的部分框的融合。

        Args:
            boxes: (N, 4) 检测框数组 [x1, y1, x2, y2]
            scores: (N,) 置信度数组
            class_ids: (N,) 类别ID数组
            iou_thr: IoU 阈值，高于此值的框被认为是同一目标

        Returns:
            fused_boxes: (M, 4) 融合后的框
            fused_scores: (M,) 融合后的置信度
            fused_class_ids: (M,) 融合后的类别ID
        """
        if len(boxes) == 0:
            return boxes, scores, class_ids

        boxes = np.asarray(boxes, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        class_ids = np.asarray(class_ids, dtype=np.int64)

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        class_ids = class_ids[order]

        fused_boxes = []
        fused_scores = []
        fused_class_ids = []
        used = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if used[i]:
                continue

            cluster_boxes = [boxes[i]]
            cluster_scores = [scores[i]]
            cluster_class = class_ids[i]
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j] or class_ids[j] != cluster_class:
                    continue

                xx1 = max(boxes[i, 0], boxes[j, 0])
                yy1 = max(boxes[i, 1], boxes[j, 1])
                xx2 = min(boxes[i, 2], boxes[j, 2])
                yy2 = min(boxes[i, 3], boxes[j, 3])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
                iou = inter / (area_i + area_j - inter + 1e-16)

                if iou >= iou_thr:
                    cluster_boxes.append(boxes[j])
                    cluster_scores.append(scores[j])
                    used[j] = True

            cluster_boxes = np.array(cluster_boxes)
            cluster_scores = np.array(cluster_scores)

            total_weight = cluster_scores.sum()
            fused_box = (cluster_boxes * cluster_scores[:, None]).sum(axis=0) / total_weight
            fused_score = cluster_scores.mean()

            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_class_ids.append(cluster_class)

        if len(fused_boxes) == 0:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])

        return np.array(fused_boxes), np.array(fused_scores), np.array(fused_class_ids)

    @staticmethod
    def draw_detections(img, all_boxes, class_names, colors=None, thickness=2):
        """
        在原图上绘制所有子图的检测框

        Args:
            img: 原图 (numpy 数组)，直接在原图上绘制 (原地修改)
            all_boxes: 所有检测框的列表，每个框格式为 [x1, y1, x2, y2, conf, class_id]
            class_names: 类别名列表，class_names[class_id] → 类别名
            colors: 颜色列表，不传则自动生成
            thickness: 框线宽度

        Returns:
            img (原地修改后的原图)
        """
        if colors is None:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
                      (0, 255, 255), (255, 0, 255), (128, 0, 255), (255, 128, 0)]

        for box in all_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            conf = box[4]
            cls_id = int(box[5])
            color = colors[cls_id % len(colors)]
            label = f"{class_names[cls_id]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img


# ============================================================
# 示例用法
# ============================================================
# if __name__ == "__main__":
#     # 输入图片
#     img_path = Path(__file__).resolve().parent / "predict_images" / "Image_20260901143437056.jpg"
#     out_dir = Path(__file__).resolve().parent / "output" / f"split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#
#     print(f"输入图片: {img_path}")
#     print(f"输出目录: {out_dir}")
#
#     # 方式1: 网格分割 (2×2)
#     print("\n--- 方式1: 网格分割 2×2 ---")
#     splits = ImageSplitter.split_by_grid(img_path, rows=2, cols=2)
#     print(f"切分为 {len(splits)} 个子图，每个尺寸: {splits[0][0].shape[:2]}")
#     ImageSplitter.save_splits(splits, out_dir / "grid")
#
#     # 方式2: 按尺寸分割 (640×640)
#     print("\n--- 方式2: 按尺寸分割 640×640 ---")
#     splits = ImageSplitter.split_by_size(img_path, tile_w=640, tile_h=640)
#     print(f"切分为 {len(splits)} 个子图")
#     ImageSplitter.save_splits(splits, out_dir / "size")
#
#     # 方式3: 带重叠分割 (640×640, 20% 重叠)
#     print("\n--- 方式3: 带重叠分割 640×640, overlap=0.2 ---")
#     splits = ImageSplitter.split_with_overlap(img_path, tile_w=640, tile_h=640, overlap=0.2)
#     print(f"切分为 {len(splits)} 个子图")
#     ImageSplitter.save_splits(splits, out_dir / "overlap")
import cv2
import numpy as np
import json
import glob
import os
from common import *

FRAME_GLOB = "cache/frames/*.png"
CRUSH_DIR = 'cache/crush-frames'
OUTPUT_PATH = "cache/00_collect.json"
GOOD_PATH = "cache/00_good.json"
MISS_PATH = "cache/00_miss.json"
THRESHOLD = 0.04 # 正常多数分布在 0.01~0.035


def evaluate_darkness(image):
    """
    综合多种策略评估图像整体偏暗程度，输出 0~1 分数。

    参数:
        image: OpenCV 读取的图像 (BGR, uint8)，或 numpy 数组

    返回:
        score: 0~1 的偏暗分数
               0.00 ~ 0.15 : 非常明亮
               0.15 ~ 0.35 : 正常/偏亮
               0.35 ~ 0.55 : 微暗/略暗
               0.55 ~ 0.75 : 明显偏暗
               0.75 ~ 0.90 : 很暗
               0.90 ~ 1.00 : 极暗/几乎全黑
        details: 各子指标详情字典

    策略混合:
        - 多颜色空间亮度分析 (Gray/HSV-V/Lab-L)
        - 直方图暗部分布统计
        - 暗区域空间集中度（避免剪影误判）
        - Sigmoid 非线性映射增强区分度
    """
    if image is None or image.size == 0:
        return 0.0, {}

    # 数据类型统一为 uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    h, w = image.shape[:2]
    total_pixels = h * w
    if total_pixels == 0:
        return 0.0, {}

    scores = {}
    weights = {}

    # ========== 1. 灰度平均亮度 ==========
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_gray = np.mean(gray)
    scores['gray_mean'] = 1.0 - (mean_gray / 255.0)
    weights['gray_mean'] = 0.10

    # ========== 2. 直方图暗部比例 (0-50) ==========
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    dark_ratio = np.sum(hist[:50]) / total_pixels
    scores['dark_ratio'] = dark_ratio
    weights['dark_ratio'] = 0.20

    # ========== 3. 直方图重心偏移 ==========
    hist_norm = hist / total_pixels
    centroid = np.sum(np.arange(256) * hist_norm) / 255.0
    scores['hist_centroid'] = 1.0 - centroid
    weights['hist_centroid'] = 0.10

    # ========== 4. HSV V通道 ==========
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_v = np.mean(v_channel)
    scores['hsv_v'] = 1.0 - (mean_v / 255.0)
    weights['hsv_v'] = 0.10

    # ========== 5. Lab L通道（人眼感知亮度） ==========
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_l = np.mean(l_channel)
    scores['lab_l'] = 1.0 - (mean_l / 255.0)
    weights['lab_l'] = 0.20

    # ========== 6. 极暗像素比例 (0-30) ==========
    very_dark_ratio = np.sum(hist[:30]) / total_pixels
    scores['very_dark_ratio'] = very_dark_ratio
    weights['very_dark_ratio'] = 0.15

    # ========== 7. 暗区域集中度（避免剪影误判） ==========
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    dark_mask = dark_mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_mask, connectivity=8)
    dark_regions = max(num_labels - 1, 0)
    dark_pixels = np.sum(dark_mask > 0)

    if dark_pixels > 0 and dark_regions > 0:
        avg_region_size = dark_pixels / dark_regions
        concentration = min(avg_region_size / (total_pixels * 0.5), 1.0)
    else:
        concentration = 0.0
    scores['dark_concentration'] = concentration
    weights['dark_concentration'] = 0.10

    # ========== 8. 最大暗区域占比 ==========
    if num_labels > 1:
        max_region_area = max(stats[1:, cv2.CC_STAT_AREA])
        max_region_ratio = max_region_area / total_pixels
    else:
        max_region_ratio = 0.0
    scores['max_dark_region'] = max_region_ratio
    weights['max_dark_region'] = 0.05

    # ========== 加权混合 + Sigmoid 映射 ==========
    total_weight = sum(weights.values())
    blended_score = sum(scores[k] * weights[k] for k in scores) / total_weight

    # Sigmoid: 增强中等亮度区域的区分度
    final_score = 1.0 / (1.0 + np.exp(-8 * (blended_score - 0.5)))

    return float(np.clip(final_score, 0.0, 1.0)), scores


def crush(input_path, output_path) -> np.ndarray:
    image = cv2.imread(input_path)
    h, w = image.shape[:2]
    roi = load_roi('layout', w, h)

    simg = crop_image(image, roi['stats'])
    timg = crop_image(image, roi['title'])
    timg = cv2.rotate(timg, cv2.ROTATE_90_CLOCKWISE)

    crushed = concat_image([timg, simg])

    if output_path:
        cv2.imwrite(output_path, crushed)

    return crushed


def process():
    # layout = load_roi_layout(ROI_PATH)
    frames = sorted(glob.glob(FRAME_GLOB))
    logger.info(f"collect start: found {len(frames)} frames.")

    good = ClusterTable()
    miss = ClusterTable()
    # clusters = [{"step": "retain", "frames": []},
    #             {"step": "abandon", "frames": []}]

    os.makedirs(CRUSH_DIR, exist_ok=True)
    t = Sheet('00_collect')
    for path in frames:
        name = os.path.basename(path)
        cimg = crush(path, os.path.join(CRUSH_DIR, name))
        score, details = evaluate_darkness(cimg)
        c = good if score <= THRESHOLD else miss
        c.add_frame(name)

        logger.debug(f'{name} {score=} {details=}')
        t.insert({'ts':os.path.splitext(name)[0], 'score':score, **details})
    t.close()


    # for i in range(2):
    #     clusters[i]['cluster_id'] = i
    #     clusters[i]['max_laplacian_var'] = 0
    #     clusters[i]['low_quality'] = False
    #     if clusters[i]['frames']:
    #         clusters[i]['representative_frame'] = clusters[0]['frames'][0]

    os.makedirs(os.path.dirname(GOOD_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MISS_PATH), exist_ok=True)
    good.save(GOOD_PATH)
    miss.save(MISS_PATH)
    # with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    #     json.dump(clusters, f, ensure_ascii=False, indent=2)

    logger.info(f"collect done: saved -> {GOOD_PATH}; {MISS_PATH}")


# ==================== 使用示例 ====================
if __name__ == '__main__':
    process()

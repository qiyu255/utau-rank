#!/usr/bin/env python3
"""
页面聚类脚本
根据相邻帧的 ROI 差异和 SSIM 将帧序列切分成页面簇。
"""

import json
import logging
import os
import sys
from typing import Any

import cv2
import numpy as np

# ---------------------------- 全局可调参数 ----------------------------
# ROI 权重，用于加权差异和加权 SSIM
TITLE_WEIGHT = 0.5
STATS_WEIGHT = 0.5

# 高斯模糊核大小（必须为奇数）
GAUSSIAN_KERNEL = (5, 5)

# 差异二值化阈值（像素值差值的阈值）
PIXEL_DIFF_THRESHOLD = 15

# 形态学开运算核大小（必须为奇数）
MORPH_KERNEL = (3, 3)

# SSIM 计算窗口大小（必须为奇数）
SSIM_WINDOW_SIZE = 11

# 判定阈值：同一页面需要同时满足 SSIM > 此值 且 diff < 此值
SSIM_THRESHOLD = 0.3
DIFF_THRESHOLD = 0.5

# 拉普拉斯方差阈值：若页面簇的最大拉普拉斯方差低于此值，标记为低质量
LAPLACIAN_LOW_QUALITY_THRESHOLD = 100

# 输入输出路径
FILTER_PATH = "cache/00_filter.json"
ROI_LAYOUT_PATH = "data/roi_layout.json"
FRAMES_DIR = "cache/frames"
OUTPUT_PATH = "cache/01_cluster.json"

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------- 工具函数 ----------------------------
def load_filter(path: str) -> list[dict[str, Any]]:
    """读取过滤器 JSON 文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Filter file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Filter file must contain a JSON array, got {type(data)}")
    return data


def load_roi_layout(path: str) -> dict[str, Any]:
    """读取 ROI 布局文件并只保留 stats 和 title 两个区域"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ROI layout file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        layout = json.load(f)
    boxes = layout.get("boxes", [])
    required = {"stats", "title"}
    box_dict = {box["name"]: box for box in boxes if box.get("name") in required}
    if len(box_dict) != 2:
        missing = required - box_dict.keys()
        raise ValueError(f"Missing required ROI boxes in layout: {missing}")
    return {
        "boxes": box_dict,
        "referenceWidth": layout["referenceWidth"],
        "referenceHeight": layout["referenceHeight"],
    }


def scale_roi(
    box: dict[str, int], img_width: int, img_height: int, ref_w: int, ref_h: int
) -> tuple[int, int, int, int]:
    """根据参考尺寸缩放 ROI 至实际图像尺寸，返回 (x, y, w, h)，均为整数"""
    sx = img_width / ref_w
    sy = img_height / ref_h
    x = int(round(box["x"] * sx))
    y = int(round(box["y"] * sy))
    w = int(round(box["w"] * sx))
    h = int(round(box["h"] * sy))
    return x, y, w, h


def extract_roi(gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    从灰度图中提取 ROI，自动裁剪至图像边界。
    如果有效区域为空则抛出异常。
    """
    img_h, img_w = gray.shape[:2]
    # 计算交叠区域
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"ROI ({x}, {y}, {w}, {h}) is completely outside image boundaries ({img_w}x{img_h})"
        )
    if x1 != x or y1 != y or x2 != x + w or y2 != y + h:
        logger.warning(
            "ROI (%d, %d, %d, %d) partially outside image (%dx%d), clipped to (%d, %d, %d, %d)",
            x, y, w, h, img_w, img_h, x1, y1, x2 - x1, y2 - y1,
        )
    return gray[y1:y2, x1:x2]


def read_gray(path: str) -> np.ndarray:
    """读取图片并转换为灰度图，返回 numpy 数组"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image or convert to grayscale: {path}")
    return img


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """计算图像的拉普拉斯方差"""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_difference(roi1: np.ndarray, roi2: np.ndarray) -> float:
    """
    计算两个 ROI 的差异比例。
    步骤：高斯模糊 -> 绝对差分 -> 二值化 -> 形态学开运算 -> 非零像素比例
    """
    blurred1 = cv2.GaussianBlur(roi1, GAUSSIAN_KERNEL, 0)
    blurred2 = cv2.GaussianBlur(roi2, GAUSSIAN_KERNEL, 0)
    diff = cv2.absdiff(blurred1, blurred2)
    _, thresh = cv2.threshold(diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    non_zero = cv2.countNonZero(opened)
    total = opened.size
    return non_zero / total


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    自行实现的结构相似度 (SSIM) 计算，使用高斯加权窗口。
    返回平均 SSIM 值（标量）。
    """
    # 转换为浮点型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    ksize = (SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, ksize, sigma)
    mu2 = cv2.GaussianBlur(img2, ksize, sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, ksize, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, ksize, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, ksize, sigma) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compare_frames(
    prev_roi_stats: np.ndarray,
    prev_roi_title: np.ndarray,
    curr_roi_stats: np.ndarray,
    curr_roi_title: np.ndarray,
) -> tuple[bool, float, float]:
    """
    比较两帧的 ROI，返回 (是否同一页面, 加权差异, 加权 SSIM)。
    """
    # 分别计算差异
    title_diff = compute_difference(prev_roi_title, curr_roi_title)
    stats_diff = compute_difference(prev_roi_stats, curr_roi_stats)
    weighted_diff = TITLE_WEIGHT * title_diff + STATS_WEIGHT * stats_diff

    # 分别计算 SSIM
    title_ssim = compute_ssim(prev_roi_title, curr_roi_title)
    stats_ssim = compute_ssim(prev_roi_stats, curr_roi_stats)
    weighted_ssim = TITLE_WEIGHT * title_ssim + STATS_WEIGHT * stats_ssim

    is_same = (weighted_ssim > SSIM_THRESHOLD) and (weighted_diff < DIFF_THRESHOLD)
    return is_same, weighted_diff, weighted_ssim


# ---------------------------- 聚类主逻辑 ----------------------------
def process_filter_cluster(
    filter_cluster: dict[str, Any],
    roi_boxes: dict[str, dict[str, int]],
    ref_w: int,
    ref_h: int,
    start_cluster_id: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    处理单个 filter cluster，将其 frames 切分成多个页面簇。
    返回 (生成的页面簇列表, 下一个可用 cluster_id)。
    """
    frames: list[str] = filter_cluster.get("frames", [])
    if not frames:
        logger.warning("filter cluster %s has empty frames, skipping", filter_cluster.get("cluster_id"))
        return [], start_cluster_id

    clusters: list[dict[str, Any]] = []
    current_cluster_id = start_cluster_id

    # 状态变量：上一个有效帧的 ROI 和当前页面簇信息
    prev_roi_stats: np.ndarray | None = None
    prev_roi_title: np.ndarray | None = None
    current_cluster: dict[str, Any] | None = None

    for idx, fname in enumerate(frames):
        img_path = os.path.join(FRAMES_DIR, fname)
        logger.info("Processing frame %s", fname)

        # 读取当前帧
        try:
            gray = read_gray(img_path)
        except Exception as e:
            logger.error("Skipping frame %s due to error: %s", fname, e)
            continue  # 跳过，可能影响连续性，但记录日志

        # 当前帧拉普拉斯方差
        lap_var = compute_laplacian_variance(gray)

        # 提取 stats 和 title 两个 ROI
        img_h, img_w = gray.shape[:2]
        try:
            stats_box = roi_boxes["stats"]
            title_box = roi_boxes["title"]
            sx, sy, sw, sh = scale_roi(stats_box, img_w, img_h, ref_w, ref_h)
            tx, ty, tw, th = scale_roi(title_box, img_w, img_h, ref_w, ref_h)
            curr_roi_stats = extract_roi(gray, sx, sy, sw, sh)
            curr_roi_title = extract_roi(gray, tx, ty, tw, th)
        except Exception as e:
            logger.error("ROI extraction failed for %s: %s", fname, e)
            continue

        if idx == 0 or current_cluster is None:
            # 第一帧或需要新建页面簇
            # 结束上一个簇（如果存在）
            if current_cluster is not None:
                max_var = current_cluster["max_laplacian_var"]
                current_cluster["low_quality"] = max_var < LAPLACIAN_LOW_QUALITY_THRESHOLD
                clusters.append(current_cluster)
                current_cluster_id += 1

            # 开始新簇
            current_cluster = {
                "cluster_id": current_cluster_id,
                "step": "cluster",
                "max_laplacian_var": lap_var,
                "low_quality": False,  # 占位，结束时更新
                "representative_frame": fname,
                "frames": [fname],
            }
            # 更新 prev ROI
            prev_roi_stats = curr_roi_stats
            prev_roi_title = curr_roi_title
            continue

        # 与上一帧比较
        assert prev_roi_stats is not None and prev_roi_title is not None
        is_same, w_diff, w_ssim = compare_frames(
            prev_roi_stats, prev_roi_title, curr_roi_stats, curr_roi_title
        )
        decision = "SAME" if is_same else "NEW"
        logger.info(
            "  weighted_diff=%.4f, weighted_ssim=%.4f -> %s",
            w_diff, w_ssim, decision,
        )

        if is_same:
            # 属于同一页面簇
            current_cluster["frames"].append(fname)
            current_cluster["max_laplacian_var"] = max(
                current_cluster["max_laplacian_var"], lap_var
            )
            # 更新 prev ROI 为当前帧
            prev_roi_stats = curr_roi_stats
            prev_roi_title = curr_roi_title
        else:
            # 结束当前簇，开始新簇
            max_var = current_cluster["max_laplacian_var"]
            current_cluster["low_quality"] = max_var < LAPLACIAN_LOW_QUALITY_THRESHOLD
            clusters.append(current_cluster)
            current_cluster_id += 1

            # 新建页面簇，以当前帧为第一帧
            current_cluster = {
                "cluster_id": current_cluster_id,
                "step": "cluster",
                "max_laplacian_var": lap_var,
                "low_quality": False,
                "representative_frame": fname,
                "frames": [fname],
            }
            prev_roi_stats = curr_roi_stats
            prev_roi_title = curr_roi_title

    # 循环结束，收尾最后一个簇
    if current_cluster is not None:
        max_var = current_cluster["max_laplacian_var"]
        current_cluster["low_quality"] = max_var < LAPLACIAN_LOW_QUALITY_THRESHOLD
        clusters.append(current_cluster)
        current_cluster_id += 1

    return clusters, current_cluster_id


def save_clusters(clusters: list[dict[str, Any]], path: str) -> None:
    """将聚类结果写入 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d page clusters to %s", len(clusters), path)


# ---------------------------- 主入口 ----------------------------
def main() -> None:
    # 加载配置
    filter_list = load_filter(FILTER_PATH)
    roi_layout = load_roi_layout(ROI_LAYOUT_PATH)
    roi_boxes = roi_layout["boxes"]  # dict with "stats" and "title"
    ref_w = roi_layout["referenceWidth"]
    ref_h = roi_layout["referenceHeight"]

    all_clusters: list[dict[str, Any]] = []
    next_cluster_id = 0

    # for fc in filter_list:
    #     logger.info(
    #         "Processing filter cluster_id=%s with %d frames",
    #         fc.get("cluster_id"), len(fc.get("frames", [])),
    #     )
    clusters, next_cluster_id = process_filter_cluster(
        filter_list[0], roi_boxes, ref_w, ref_h, next_cluster_id
    )
    all_clusters.extend(clusters)

    save_clusters(all_clusters, OUTPUT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)
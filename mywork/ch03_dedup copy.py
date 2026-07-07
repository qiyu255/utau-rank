import cv2
import numpy as np
from pathlib import Path
from common import *

TASK_NAME = Path(__file__).stem

# ---------- 全局阈值 ----------
SSIM_THRESHOLD = 0.85
DIFF_THRESHOLD = 0.02
GAUSS_KERNEL = (5, 5)
MORPH_KERNEL = (5, 5)
THRESH_VALUE = 30
# -----------------------------

def calc_ssim(g1: np.ndarray, g2: np.ndarray) -> float:
    """自行实现 SSIM，不依赖 skimage"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel)
    mu1 = cv2.filter2D(g1, -1, window)
    mu2 = cv2.filter2D(g2, -1, window)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(g1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(g2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(g1 * g2, -1, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))

def calc_diff(g1: np.ndarray, g2: np.ndarray) -> float:
    """加权差异：高斯模糊 -> absdiff -> 二值化 -> 开运算 -> 非零像素占比"""
    b1 = cv2.GaussianBlur(g1, GAUSS_KERNEL, 0)
    b2 = cv2.GaussianBlur(g2, GAUSS_KERNEL, 0)
    diff = cv2.absdiff(b1, b2)
    _, th = cv2.threshold(diff, THRESH_VALUE, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    total = opened.shape[0] * opened.shape[1]
    nonzero = cv2.countNonZero(opened)
    return nonzero / total

def get_quality(img_bgr: np.ndarray) -> float:
    """拉普拉斯方差作为质量指标"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def cluster_pages(image_paths: list[Path]):
    N = len(image_paths)
    if N == 0:
        return [], None, None

    name_to_path = {p.name: p for p in image_paths}
    stats_records = []               # 每帧记录：[quality, ssim, diff, cluster_id, is_new]
    clusters = []
    curr_cluster = None
    ref_img = None
    ref_q = 0.0
    cid = 0                          # cluster id counter

    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        q = get_quality(img)

        if i == 0:   # 第一帧：建立第一个簇
            cid = 1
            curr_cluster = {
                "id": cid,
                "best": path.name,
                "frames": [path.name],
                "indicator": {"quality": q}
            }
            ref_img = gray.copy()
            ref_q = q
            stats_records.append([q, 1.0, 0.0, cid, 1])
            continue

        # 与当前簇参考图比较
        ssim_val = calc_ssim(ref_img, gray)
        diff_val = calc_diff(ref_img, gray)
        same_page = (ssim_val > SSIM_THRESHOLD) and (diff_val < DIFF_THRESHOLD)

        if same_page:
            curr_cluster["frames"].append(path.name)
            if q > ref_q:
                curr_cluster["best"] = path.name
                curr_cluster["indicator"]["quality"] = q
                ref_img = gray.copy()
                ref_q = q
            stats_records.append([q, ssim_val, diff_val, curr_cluster["id"], 0])
        else:
            clusters.append(curr_cluster)
            cid += 1
            curr_cluster = {
                "id": cid,
                "best": path.name,
                "frames": [path.name],
                "indicator": {"quality": q}
            }
            ref_img = gray.copy()
            ref_q = q
            stats_records.append([q, ssim_val, diff_val, cid, 1])

    # 最后一个簇
    if curr_cluster is not None:
        clusters.append(curr_cluster)

    # 构建统计矩阵
    stats_arr = np.array(stats_records, dtype=np.float64)
    frame_names = [p.name for p in image_paths]
    columns = ["quality", "ssim", "diff", "cluster_id", "is_new"]

    # 收集最优帧路径
    best_paths = [name_to_path[c["best"]] for c in clusters]
    return best_paths, clusters, stats_arr, frame_names, columns

def run(image_paths: list[Path], ctx: Context):
    N = len(image_paths)
    logger.info(f"%s start: input %d frames.", TASK_NAME, N)

    best_paths, clusters, stats_arr, frame_names, columns = cluster_pages(image_paths)

    # 保存统计数据
    store.stats(TASK_NAME + '_page_stats', stats_arr, frame_names, columns)
    # 保存簇信息
    store.clusters(TASK_NAME + '_clusters', clusters)

    logger.info(f"%s done:  output %d frames.", TASK_NAME, len(best_paths))
    return best_paths
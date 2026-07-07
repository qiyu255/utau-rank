import cv2
import numpy as np
from pathlib import Path
from common import *
TASK_NAME = Path(__file__).stem
# ============ Global Thresholds ============
SSIM_THRESHOLD = 0.97
DIFF_THRESHOLD = 0.03
BLUR_KERNEL = (3, 3)
DIFF_BIN_THRESH = 25
MORPH_KERNEL = 1
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
QUALITY_WEIGHTS = np.array([0.4, 0.2, 0.2, 0.2])
QUALITY_FIELDS = ['score', 'sharpness', 'brightness', 'contrast', 'entropy']
COMPARE_FIELDS = ['weighted_ssim', 'weighted_diff', 'same_page']


def gaussian_window() -> np.ndarray:
    k = cv2.getGaussianKernel(SSIM_WINDOW, SSIM_SIGMA)
    return np.outer(k, k.T)


def ssim(img1, img2, window) -> np.float64:
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    pad = SSIM_WINDOW // 2
    mu1 = cv2.filter2D(i1, -1, window)[pad:-pad, pad:-pad]
    mu2 = cv2.filter2D(i2, -1, window)[pad:-pad, pad:-pad]
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(i1 * i1, -1, window)[pad:-pad, pad:-pad] - mu1_sq
    sigma2_sq = cv2.filter2D(i2 * i2, -1, window)[pad:-pad, pad:-pad] - mu2_sq
    sigma12 = cv2.filter2D(i1 * i2, -1, window)[pad:-pad, pad:-pad] - mu1_mu2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return (num / den).mean()


def diff_score(img1, img2) -> np.float64:
    b1 = cv2.GaussianBlur(img1, BLUR_KERNEL, 0)
    b2 = cv2.GaussianBlur(img2, BLUR_KERNEL, 0)
    d = cv2.absdiff(b1, b2)
    _, t = cv2.threshold(d, DIFF_BIN_THRESH, 255, cv2.THRESH_BINARY)
    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    m = cv2.morphologyEx(t, cv2.MORPH_OPEN, k)
    return (np.count_nonzero(m)) / m.size


def quality_metrics(gray) -> tuple:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = float(lap.var()) / 1000.0
    bright = float(gray.mean()) / 255.0
    bright_score = 1.0 - abs(bright - 0.5) * 2.0
    contrast = float(gray.std()) / 128.0
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum()) / 8.0
    raw = np.clip([sharp, bright_score, contrast, entropy], 0.0, 1.0)
    score = 1.0 / (1.0 + np.exp(-6.0 * (np.dot(raw, QUALITY_WEIGHTS) - 0.5)))
    return score, sharp, bright, contrast, entropy


def add_cluster(clusters, path, score):
    c = {'id': len(clusters) + 1, 'best': path.name,
         'frames': [path.name], 'indicator': {'quality': score}}
    clusters.append(c)
    return c


def cluster_frames(image_paths, window):
    clusters = []
    quality_arr = np.zeros((len(image_paths), 5), dtype=np.float64)
    compare_arr = np.zeros((len(image_paths), 3), dtype=np.float64)
    current = None
    best_gray = None
    best_quality = -1.0
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = quality_metrics(gray)
        quality_arr[i] = m
        score = m[0]
        if current is None:
            current = add_cluster(clusters, path, score)
            best_gray, best_quality = gray, score
            compare_arr[i] = (1.0, 0.0, 1)
            continue
        s = ssim(best_gray, gray, window)
        d = diff_score(best_gray, gray)
        same = int(s > SSIM_THRESHOLD and d < DIFF_THRESHOLD)
        compare_arr[i] = (s, d, same)
        if not same:
            current = add_cluster(clusters, path, score)
            best_gray, best_quality = gray, score
            continue
        current['frames'].append(path.name)
        if score > best_quality:
            best_quality, best_gray = score, gray
            current['best'] = path.name
            current['indicator']['quality'] = score
    return clusters, quality_arr, compare_arr


def run(image_paths: list[Path], ctx: Context):
    N = len(image_paths)
    logger.info(f"%s start: input %d frames.", TASK_NAME, N)
    window = gaussian_window()
    clusters, quality_arr, compare_arr = cluster_frames(image_paths, window)
    names = [p.name for p in image_paths]
    store.stats(TASK_NAME + '_quality', quality_arr, names, QUALITY_FIELDS)
    store.stats(TASK_NAME + '_compare', compare_arr, names, COMPARE_FIELDS)
    store.clusters(TASK_NAME, clusters)
    best_names = {c['best'] for c in clusters}
    result = [p for p in image_paths if p.name in best_names]
    logger.info(f"%s done:  output %d frames.", TASK_NAME, len(result))
    return result

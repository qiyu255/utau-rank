import cv2
import numpy as np
from pathlib import Path
from common import *
TASK_NAME = Path(__file__).stem
SAME_THRESH = 0.93
DIFF_WEIGHT = 0.1
SSIM_WEIGHT = 0.9
BLUR_KERNEL = (3, 3)
DIFF_BIN_THRESH = 25
MORPH_KERNEL = 1
SSIM_WINDOW = 3
SSIM_SIGMA = 1.5
CLOUMNS = ['sim', 'diff_sim', 'absdiff',
           'ssim', 'l', 'c', 's', 'mu', 'sigma']


def gaussian_window() -> np.ndarray:
    k = cv2.getGaussianKernel(SSIM_WINDOW, SSIM_SIGMA)
    return np.outer(k, k.T)


def ssim(rst, img1, img2, window):
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    c3 = c2 / 2  # 29.26125
    pad = SSIM_WINDOW // 2
    # 局部均值计算
    mu1 = cv2.filter2D(i1, -1, window)[pad:-pad, pad:-pad]
    mu2 = cv2.filter2D(i2, -1, window)[pad:-pad, pad:-pad]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    # 局部方差与协方差计算
    sigma1_sq = cv2.filter2D(i1 * i1, -1, window)[pad:-pad, pad:-pad] - mu1_sq
    sigma2_sq = cv2.filter2D(i2 * i2, -1, window)[pad:-pad, pad:-pad] - mu2_sq
    sigma12 = cv2.filter2D(i1 * i2, -1, window)[pad:-pad, pad:-pad] - mu1_mu2
    # 计算局部标准差，使用 maximum 避免 $sigma^2$ 浮点误差导致的负数开根号问题
    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0))
    # 三个分量：亮度、对比度、结构
    luminance = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    contrast = (2 * sigma1 * sigma2 + c2) / (sigma1_sq + sigma2_sq + c2)
    structure = (sigma12 + c3) / (sigma1 * sigma2 + c3)
    ssim_map = luminance * contrast * structure
    rst[0] = ssim_map.mean()                      # 结构相似度均值
    rst[1] = luminance.mean()                     # 亮度相似度均值
    rst[2] = contrast.mean()                      # 对比度相似度均值
    rst[3] = structure.mean()                     # 结构相似度均值
    rst[4] = np.abs(mu1 - mu2).mean() / 255       # 局部均值差异
    rst[5] = np.abs(sigma1_sq - sigma2_sq).mean()  # 局部方差差异


def diff(rst, img1, img2):
    b1 = cv2.GaussianBlur(img1, BLUR_KERNEL, 0)
    b2 = cv2.GaussianBlur(img2, BLUR_KERNEL, 0)
    d = cv2.absdiff(b1, b2)
    _, t = cv2.threshold(d, DIFF_BIN_THRESH, 255, cv2.THRESH_BINARY)
    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    m = cv2.morphologyEx(t, cv2.MORPH_OPEN, k)

    rst[0] = 1-np.count_nonzero(m) / m.size
    rst[1] = d.mean() / 255


def analyze(image_paths, window) -> np.ndarray:
    stats = np.empty((len(image_paths), 9), dtype=np.float64)
    stats[0] = np.zeros(9, dtype=np.float64)
    last = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    for i in range(1, len(image_paths)):
        current = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        diff(stats[i, 1:3], current, last)
        ssim(stats[i, 3:], current, last, window)
        stats[i][0] = DIFF_WEIGHT * stats[i][1] + SSIM_WEIGHT * stats[i][3]
        last = current
    return stats


def masking(stats):
    return stats[:, 0] < SAME_THRESH


def add_cluster(clusters, path, score):
    c = {'id': len(clusters) + 1, 'best': path.name,
         'frames': [path.name], 'indicator': {'quality': score}}
    clusters.append(c)
    return c


def cluster(data, breaks):
    a = []
    for p, b in zip(data, breaks):
        if b or not a:
            a.append([p])
        else:           
            a[-1].append(p)
    return a

def run(image_paths: list[Path], ctx: Context):
    N = len(image_paths)
    logger.info(f"%s start: input %d frames.", TASK_NAME, N)
    ids = [i.stem for i in image_paths]
    names = [i.name for i in image_paths]

    stats = analyze(image_paths, gaussian_window())
    mask = masking(stats)
    clus = cluster(image_paths, mask)
    store.stats(TASK_NAME, stats, ids, CLOUMNS)
    store.clusters(TASK_NAME, clus)
    logger.info(f"%s done:  output %d clusters.", TASK_NAME, len(clus))
    return clus

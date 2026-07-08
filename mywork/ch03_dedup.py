from numpy.lib.stride_tricks import sliding_window_view
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
CLOUMNS = ['sim', 'psim', 'absdiff',
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


def psim(rst, img1, img2):
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
        psim(stats[i, 1:3], current, last)
        ssim(stats[i, 3:], current, last, window)
        stats[i][0] = DIFF_WEIGHT * stats[i][1] + SSIM_WEIGHT * stats[i][3]
        last = current
    return stats


def masking(stats):
    return stats[:, 0] < SAME_THRESH


def slicing(mask, include_leading=False):
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]   # 所有 True 的索引

    # 处理开头：如果要求包含前导段且第一个 True 不在 0 位置
    if include_leading and idx[0] != 0:
        idx = np.r_[0, idx]   # 在开头插入 0

    # 计算每个段的结束位置（下一个 True 的位置，或数组末尾）
    starts = idx
    ends = np.r_[idx[1:], len(mask)]

    return [slice(s, e) for s, e in zip(starts, ends)]


def find_best_plateau(data, window_size=5, extend=False):
    """
    寻找唯一一个方差最小（最平坦）的区间
    返回: (start, end) 闭区间索引
    """
    n = len(data)
    if n < window_size:
        return None
    # 1. 计算滑动窗口方差
    sw = sliding_window_view(data, window_size)
    variances = np.var(sw, axis=1, ddof=1)
    # 2. 找到方差最小的窗口索引
    best_idx = np.argmin(variances)
    # 3. 提取该窗口的起始和结束索引 (闭区间)
    start = best_idx
    end = best_idx + window_size - 1
    # 4. (可选) 以此窗口为基准，向两侧延伸，捕获完整的平台期
    if extend:
        mean_val = np.mean(data[start:end+1])
        tol = np.std(data[start:end+1]) * 3  # 容忍度：基于该窗口的标准差
        # 向左扩展
        while start > 0 and abs(data[start - 1] - mean_val) < tol:
            start -= 1
        # 向右扩展
        while end < n - 1 and abs(data[end + 1] - mean_val) < tol:
            end += 1
    return start, end


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
    slices = slicing(mask)
    best_indexs = []
    absdiff = stats[:, 2]
    for s in slices:
        logger.debug('find_best_plateau: %s', absdiff[s])
        i = find_best_plateau(absdiff[s], 5)[1]
        pi = s.start + i
        best_indexs.append(pi)
        logger.debug('best_plateau: %d -> %d -> %s', i, pi, image_paths[pi])

    # clus = cluster(image_paths, mask)
    store.stats(TASK_NAME, stats, ids, CLOUMNS)
    store.clusters(TASK_NAME, names, slices, best_indexs)
    logger.info(f"%s done:  output %d frames.", TASK_NAME, len(best_indexs))
    return [image_paths[i] for i in best_indexs]

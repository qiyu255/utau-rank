from dataclasses import dataclass, field
from collections import defaultdict
from itertools import compress
import numpy as np
from pathlib import Path
from common import *

TASK_NAME = Path(__file__).stem
MASK_BY = 0.03445
FIDLE_NAMES = ['cost',
               'gray_mean',
               'dark_ratio',
               'hist_centroid',
               'hsv_v',
               'lab_l',
               'very_dark_ratio',
               'dark_concentration',
               'max_dark_region'
               ]
WEIGHTS = np.array([0.10, 0.20, 0.10, 0.10, 0.20, 0.15, 0.10, 0.05])

def masking(stats) -> np.ndarray:
    return stats[:, 0] <= MASK_BY


def scan(image_paths) -> np.ndarray:
    stats = np.empty((len(image_paths), 9), dtype=np.float64)
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        total = h * w

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_g = np.mean(gray)
        stats[i, 1] = 1.0 - (mean_g / 255.0)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        stats[i, 2] = np.sum(hist[:50]) / total

        hist_norm = hist / total
        centroid = np.sum(np.arange(256) * hist_norm) / 255.0
        stats[i, 3] = 1.0 - centroid

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_ch = hsv[:, :, 2]
        stats[i, 4] = 1.0 - (np.mean(v_ch) / 255.0)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]
        stats[i, 5] = 1.0 - (np.mean(l_ch) / 255.0)

        stats[i, 6] = np.sum(hist[:30]) / total

        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        dark_mask = dark_mask.astype(np.uint8)
        n_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        dark_pixels = np.sum(dark_mask > 0)
        dark_regions = max(n_labels - 1, 0)
        if dark_pixels > 0 and dark_regions > 0:
            avg_size = dark_pixels / dark_regions
            stats[i, 7] = min(avg_size / (total * 0.5), 1.0)
        else:
            stats[i, 7] = 0.0

        if n_labels > 1:
            max_area = max(cc_stats[1:, cv2.CC_STAT_AREA])
            stats[i, 8] = max_area / total
        else:
            stats[i, 8] = 0.0

    blended = np.dot(stats[:, 1:], WEIGHTS)
    stats[:, 0] = 1.0 / (1.0 + np.exp(-8 * (blended - 0.5)))
    return stats

def run(image_paths: list[Path], ctx: Context):
    N = len(image_paths)
    logger.info(f"%s start: input %d frames.", TASK_NAME, N)

    stats = scan(image_paths)
    store.stats(TASK_NAME, stats, [i.stem for i in image_paths], FIDLE_NAMES)
    mask = masking(stats)
    good = list(compress(image_paths, mask))
    bad = list(compress(image_paths, ~mask))
    store.entries(TASK_NAME+'_good', [i.name for i in good])
    store.entries(TASK_NAME+'_bad', [i.name for i in bad])

    logger.info(f"%s done:  output %d frames.", TASK_NAME, N)
    return good
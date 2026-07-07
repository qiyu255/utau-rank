import cv2
import numpy as np
from itertools import compress
from pathlib import Path
from common import *
TASK_NAME = Path(__file__).stem
MASK_BY = 0.25
FIDLE_NAMES = ['cost',
               'concentration',
               'h_cont',
               'peak',
               'bg_purity',
               'dispersion',
               'band_ratio']


def extract_bands(mask: np.ndarray) -> list:
    if not np.any(mask):
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))
    return list(zip(starts, ends))


def max_ones(arr: np.ndarray) -> int:
    if len(arr) == 0:
        return 0
    bands = extract_bands(arr)
    if not bands:
        return int(arr[0]) * len(arr)
    return max(e - s for s, e in bands)


def eval_roi(roi: np.ndarray) -> tuple:
    h, w = roi.shape
    _, binary = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_fg = np.sum(binary == 255, axis=1) / max(w, 1)
    peak_val = float(np.max(row_fg))
    above = row_fg > peak_val * 0.3
    bands = extract_bands(above)
    main_band = max(bands, key=lambda b: b[1] - b[0]) if bands else (0, 1)
    band_h = main_band[1] - main_band[0]
    band_ratio = band_h / max(h, 1)
    total_mass = np.sum(row_fg)
    if total_mass > 0.01:
        concentration = np.sum(row_fg[main_band[0]:main_band[1]]) / total_mass
    else:
        concentration = 0.0
    cy = min(max((main_band[0] + main_band[1]) // 2, 0), h - 1)
    h_cont = max_ones(binary[cy] == 255) / max(w, 1)
    outside = np.concatenate([row_fg[:main_band[0]], row_fg[main_band[1]:]])
    outside_max = float(np.max(outside)) if len(outside) > 0 else 0.0
    bg_purity = (1.0 - outside_max) * (1.0 - max(band_ratio - 0.5, 0.0) / 0.5)
    fg_idx = np.where(row_fg > 0.05)[0]
    dispersion = 0.0
    if len(fg_idx) > 1:
        span = fg_idx[-1] - fg_idx[0] + 1
        dispersion = min((span / len(fg_idx) - 1.0) * 0.5, 1.0)
    c_score = 1.0 - concentration
    h_score = 1.0 - h_cont
    p_score = 1.0 - peak_val
    b_score = 1.0 - bg_purity
    d_score = dispersion
    r_score = min(abs(band_ratio - 0.35) / 0.5, 1.0)
    cost = (c_score * 0.20 + h_score * 0.20 + p_score * 0.15 +
            b_score * 0.20 + d_score * 0.15 + r_score * 0.10)
    return (cost, concentration, h_cont, peak_val, bg_purity, dispersion, band_ratio)


def masking(stats: np.ndarray) -> np.ndarray:
    return stats[:, 0] <= MASK_BY


def scan(image_paths) -> np.ndarray:
    stats = np.empty((len(image_paths), len(FIDLE_NAMES)), dtype=np.float64)
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        boxes = get_boxes('line', img.shape[1], img.shape[0])
        best_cost = -1.0
        best_feats = None
        for box in boxes.values():
            roi = crop(img, box)
            if roi.shape[0] > roi.shape[1]:
                roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            if roi.ndim == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            feats = eval_roi(roi)
            if feats[0] > best_cost:
                best_cost = feats[0]
                best_feats = feats
        stats[i] = best_feats
    return stats


def run(image_paths: list[Path], ctx: Context):
    N = len(image_paths)
    logger.info(f"%s start: input %d frames.", TASK_NAME, N)
    stats = scan(image_paths)
    store.stats(TASK_NAME, stats, [p.stem for p in image_paths], FIDLE_NAMES)
    mask = masking(stats)
    good = list(compress(image_paths, mask))
    bad = list(compress(image_paths, ~mask))
    store.entries(TASK_NAME + '_good', [p.name for p in good])
    store.entries(TASK_NAME + '_bad', [p.name for p in bad])
    logger.info(f"%s done:  output %d frames.", TASK_NAME, N)
    return good

import cv2
import numpy as np
import json
import os
from common import *

FRAME_DIR = 'cache/crush-frames'
OUTPUT_HIT_PATH = "cache/01_sift_hit.json"
OUTPUT_MISS_PATH = "cache/01_sift_miss.json"
INPUT_PATH = "cache/00_good.json"
HIT_COST = 0.25  # 低cost阈值，需根据实际数据分布微调

# clusters = {
#     'hit': ClusterTable(),
#     'miss': ClusterTable()
# }

# _sheets = {}
# def get_sheets(key):
#     if key not in _sheets:
#         _sheets[key] = Sheet(f'01_sift_{key}')
#     return _sheets[key]


def extract_bands(mask: np.ndarray):
    """提取mask中连续True的区间，返回[(start, end), ...]"""
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


def max_consecutive_ones(arr: np.ndarray) -> int:
    """返回布尔数组中最大连续True长度"""
    if len(arr) == 0:
        return 0
    diff = np.diff(arr.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if arr[0]:
        starts = np.insert(starts, 0, 0)
    if arr[-1]:
        ends = np.append(ends, len(arr))
    if len(starts) == 0:
        return int(arr[0]) * len(arr)
    lengths = ends - starts
    return int(np.max(lengths)) if len(lengths) > 0 else 0


def evaluate_strip(img: np.ndarray) -> dict:
    """综合评分ROI图像的横条相似度，cost越低越像横条"""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # OTSU二值化，前景（深色）为255
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 行前景比例
    row_fg = np.sum(binary == 255, axis=1) / max(w, 1)

    # 找主峰行
    peak_idx = int(np.argmax(row_fg))
    peak_val = float(row_fg[peak_idx])

    # 条带范围：峰值附近超过30%峰高的连续区域
    half_val = peak_val * 0.3
    above = row_fg > half_val
    bands = extract_bands(above)
    main_band = max(
        bands, key=lambda b: b[1] - b[0], default=(peak_idx, peak_idx + 1))
    band_h = main_band[1] - main_band[0]

    # 集中度：条带内前景量 / 总前景量
    band_mass = np.sum(row_fg[main_band[0]:main_band[1]])
    total_mass = np.sum(row_fg)
    concentration = band_mass / total_mass if total_mass > 0.01 else 0.0

    # 横向连续性：条带中心行的最大连续前景段占比
    cy = (main_band[0] + main_band[1]) // 2
    cy = min(max(cy, 0), h - 1)
    center_row = binary[cy]
    h_cont = max_consecutive_ones(center_row == 255) / max(w, 1)

    # 条带占比
    band_ratio = band_h / max(h, 1)

    # 背景纯度：条带外应干净，条带占满时强制为0
    outside_vals = []
    if main_band[0] > 0:
        outside_vals.extend(row_fg[:main_band[0]])
    if main_band[1] < h:
        outside_vals.extend(row_fg[main_band[1]:])
    outside_max = max(outside_vals) if outside_vals else 0.0
    bg_purity = (1.0 - outside_max) * (1.0 - max(band_ratio - 0.5, 0.0) / 0.5)

    # 前景分散度：前景行是否集中在单一连续带
    fg_idx = np.where(row_fg > 0.05)[0]
    dispersion = 0.0
    if len(fg_idx) > 1:
        span = fg_idx[-1] - fg_idx[0] + 1
        dispersion = min((span / len(fg_idx) - 1.0) * 0.5, 1.0)

    # 综合cost（0=最像横条，1=最不像）
    c_score = 1.0 - concentration
    h_score = 1.0 - h_cont
    p_score = 1.0 - peak_val
    b_score = 1.0 - bg_purity
    d_score = dispersion
    r_score = min(abs(band_ratio - 0.35) / 0.5, 1.0)

    cost = (
        c_score * 0.20 +
        h_score * 0.20 +
        p_score * 0.15 +
        b_score * 0.20 +
        d_score * 0.15 +
        r_score * 0.10
    )

    return {
        'cost': round(float(cost), 4),
        'concentration': round(float(concentration), 4),
        'h_cont': round(float(h_cont), 4),
        'peak': round(float(peak_val), 4),
        'bg_purity': round(float(bg_purity), 4),
        'dispersion': round(float(dispersion), 4),
        'band_ratio': round(float(band_ratio), 4),
    }


# 预处理图像，注意执行时机是在裁剪后的roi图块
def preprocess(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    # 实际支持横条或竖条
    # 统一到一个方向，这里水平，如果列处理方便可以统一到垂直
    if h > w:
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    # 转为灰度图
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # 轻微高斯模糊去噪
    im = cv2.GaussianBlur(im, (3, 3), 0)

    return im


def extract_rois(im: np.ndarray, boxes: dict) -> dict:
    """根据box字典裁剪ROI图块"""
    roi = {}
    for label, box in boxes.items():
        roi[label] = crop_image(im, box)
    return roi


# 单个图像处理流程
def process_frame_rois(roi: dict, name: str):
    details_list = []
    for label, img in roi.items():
        img = preprocess(img)
        details = evaluate_strip(img)
        details_list.append((label, details))

    # 多个roi要整体都像才算像：取最差（最大cost）
    overall_label, overall = max(details_list, key=lambda x: x[1]['cost'])
    category = 'hit' if overall['cost'] < HIT_COST else 'miss'

    ts = os.path.splitext(name)[0]
    for label, details in details_list:
        # get_sheets(label).insert()
        submit_sheet(('01_sift', category), {'ts': ts, **details})
        logger.debug(f'{name} {label} {category} {details=}')

    # clusters[category].add_frame(name)
    submit_frame(('01_sift', category), name, indicator=overall)
    # get_sheets('all').insert({'ts': os.path.splitext(name)[0], **overall})
    submit_sheet(('01_sift', 'all'), {'ts': ts, **overall})


# 单个图像加载流程
def load_frame_and_process(name: str):
    im = cv2.imread(os.path.join(FRAME_DIR, name))
    h, w = im.shape[:2]
    boxes = load_roi('line', w, h)
    roi = extract_rois(im, boxes)
    process_frame_rois(roi, name)


# 主流程
def process():
    logger.info(f"sift start: load {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        name = item['best']
        load_frame_and_process(name)

    # for sheet in _sheets.values():
    #     sheet.close()

    # clusters['hit'].save(OUTPUT_HIT_PATH)
    # clusters['miss'].save(OUTPUT_MISS_PATH)
    flush_submit()
    logger.info(f"sift done.")


if __name__ == '__main__':
    process()

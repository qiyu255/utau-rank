

import math
from functools import lru_cache
import cv2
import numpy as np
from itertools import compress
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from common import *
TASK_NAME = Path(__file__).stem
LAYOUT_DIR = Path('cache/layout')
LAYOUT_DIR.mkdir(parents=True, exist_ok=True)


def pallette_names(hsv_list):
    names = []
    for h, s, v in hsv_list:
        # 1. 低饱和度 → 灰色（阈值下调，避免淡彩被误判）
        if s < 25:
            names.append('gray')
            continue
        # 2. 粉色：仅限品红色调（170-180）
        if 170 <= h <= 180:
            names.append('pink')
        # 3. 橙色：覆盖红-橙-黄区域（0-25）
        elif 0 <= h <= 25:
            names.append('orange')
        # 4. 绿色
        elif 36 <= h <= 85:
            names.append('green')
        # 5. 蓝色
        elif 86 <= h <= 130:
            names.append('blue')
        # 6. 紫色：包含131-180范围，但粉色优先
        elif 131 <= h <= 170:
            names.append('purplu')
        # 7. 剩余小段（26-35 黄色调）归为橙色
        else:
            names.append('orange')

    return names


@lru_cache()
def get_boxes_by_theme(theme, w, h):
    if theme in ['purplu', 'green']:
        theme = 'orange'
    return get_boxes(theme, w, h)


def padding(img, px, py, bg):
    return cv2.copyMakeBorder(
        img,
        px, px, py, py,
        cv2.BORDER_CONSTANT,
        value=bg
    )


def normalize_size(images, bg):
    h = max(x.shape[0] for x in images)
    w = max(x.shape[1] for x in images)

    result = []

    for img in images:
        ih, iw = img.shape[:2]

        canvas = np.full(
            (h, w),
            bg,
            dtype=img.dtype
        )

        canvas[:ih, :iw] = img
        result.append(canvas)

    return result

# 新增一个机制 当网格填不满时， 用fg参数生成 h*w 的格子填充


def grid_layout(images, bg, fg, gap=0, cols=0):
    n = len(images)
    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if images[0].ndim == 3 else 0

    if cols == 0:
        best = (math.inf, 0, 0)
        for c in range(1, n + 1):
            r = math.ceil(n / c)
            last = n % c
            if last != 0 and last < c * 0.5:
                continue
            score = abs((c * w) / (r * h) - 1)
            if score < best[0]:
                best = (score, r, c)
        _, rows, cols = best
    else:
        rows = math.ceil(n / cols)

    tw = cols * w + (cols - 1) * gap
    th = rows * h + (rows - 1) * gap
    shape = (th, tw, channels) if channels else (th, tw)
    logger.warning(
        f'{h=} {w=} {n=} {gap=} {rows=} {cols=} {tw=} {th=} {shape=}')
    canvas = np.full(shape, bg, dtype=np.uint8)

    total_cells = rows * cols
    if total_cells > n:
        pad_shape = (h, w, channels) if channels else (h, w)
        pad_img = np.full(pad_shape, fg, dtype=np.uint8)
        images = list(images) + [pad_img] * (total_cells - n)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        y = row * (h + gap)
        x = col * (w + gap)
        canvas[y:y+h, x:x+w] = img

    return canvas


def take(p, t):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    m = {}
    for label, box in get_boxes_by_theme(t, w, h).items():
        m[label] = crop(im, box)
    return m


def run(param: tuple[list[list[Path]], list[list]], ctx: Context):
    groups, pallette_hsv = param
    themes = pallette_names(pallette_hsv)
    logger.info("%s start: input %d groups.", TASK_NAME, len(groups))
    logger.info("pallette: %s", pallette_hsv)

    labelimgs = defaultdict(list)
    labelthemes = defaultdict(list)
    for g, t in zip(groups, themes):
        logger.info('theme: %s, len: %d', t, len(g))
        if t == 'orange':
            g = g[1:]
        for p in g:
            m = take(p, t)
            for label in m:
                labelimgs[label].append(m[label])
                labelthemes[label].append(t)

    labelpaths = {}
    for label in labelimgs:
        imgs = normalize_size(labelimgs[label], 255)
        layout = grid_layout(imgs, 0, 255, 2)
        lp = LAYOUT_DIR/f'{label}.png'
        cv2.imwrite(lp, layout)
        labelpaths[label] = lp

    store.json(LAYOUT_DIR/'theme-track.json', dict(labelthemes))
    logger.info("%s done: output %d images.", TASK_NAME, len(labelpaths))
    return labelpaths, labelthemes

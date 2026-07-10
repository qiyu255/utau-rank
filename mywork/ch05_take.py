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
INFO_LABELS = ["title", "library", "author"]
DATA_LABELS = ["rank", "last", "total", "view", "comment",
                "mylist", "mylist_rate","comment_rate", "date", "id"]



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


def rander_table(images: list[list[np.ndarray | None]], bs, bc, px, py, pc):
    rows = len(images)
    cols = len(images[0])
    col_widths = [0] * cols
    row_heights = [0] * rows
    for r in range(rows):
        for c in range(cols):
            img = images[r][c]
            if img is not None:
                h, w = img.shape[:2]
                if w > col_widths[c]:
                    col_widths[c] = w
                if h > row_heights[r]:
                    row_heights[r] = h
    canvas_w = sum(col_widths) + (cols + 1) * bs + 2 * px
    canvas_h = sum(row_heights) + (rows + 1) * bs + 2 * py
    shape = hasattr(pc, '__iter__') and (
        canvas_h, canvas_w, len(pc)) or (canvas_h, canvas_w)
    canvas = np.full(shape, pc, dtype=np.uint8)
    if bs > 0:
        y = py
        for i in range(rows + 1):
            cv2.rectangle(canvas, (px, y), (canvas_w -
                          px - 1, y + bs - 1), bc, -1)
            if i < rows:
                y += bs + row_heights[i]
        x = px
        for i in range(cols + 1):
            cv2.rectangle(canvas, (x, py), (x + bs - 1,
                          canvas_h - py - 1), bc, -1)
            if i < cols:
                x += bs + col_widths[i]
    coords_map = []
    y = bs + py
    for r in range(rows):
        x = bs + px
        row_coords = []
        for c in range(cols):
            img = images[r][c]
            if img is not None:
                h, w = img.shape[:2]
                x_offset = 0
                y_offset = (row_heights[r] - h) // 2
                pos_x = x + x_offset
                pos_y = y + y_offset
                canvas[pos_y:pos_y+h, pos_x:pos_x+w] = img
                row_coords.append(
                    {'x1': pos_x, 'y1': pos_y, 'x2': pos_x + w, 'y2': pos_y + h})
            else:
                pos_x = x
                pos_y = y
                end_x = pos_x + col_widths[c]
                end_y = pos_y + row_heights[r]
                row_coords.append(
                    {'x1': pos_x, 'y1': pos_y, 'x2': end_x, 'y2': end_y})
            x += bs + col_widths[c]
        y += bs + row_heights[r]
        coords_map.append(row_coords)
    return canvas, coords_map


def take(p, t):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    m = {}
    for label, box in get_boxes_by_theme(t, w, h).items():
        if label in INFO_LABELS:
            m[label] = np.rot90(crop(im, box))
        elif label in DATA_LABELS:
            m[label] = crop(im, box)
        if label.endswith('rate'):
            m[label] = padding(m[label], 2, 2, 255)
    return m


def run(param: tuple[list[list[Path]], list[list]], ctx: Context):
    groups, pallette_hsv = param
    themes = pallette_names(pallette_hsv)
    logger.info("%s start: input %d groups.", TASK_NAME, len(groups))
    logger.info("pallette: %s", pallette_hsv)
    rows_data = []
    themes_for_rows = []
    for g, t in zip(groups, themes):
        logger.info('theme: %s, len: %d', t, len(g))
        if t == 'orange':
            g = g[1:]  # 丢弃orange的第一张
        for p in g:
            m = take(p, t)
            rows_data.append(m)
            themes_for_rows.append(t)
    info_images = []
    data_images = []
    for m in rows_data:
        title_img = m.get('title')
        h_title = title_img.shape[0] if title_img is not None else 0
        info_row = []
        data_row = []
        for label in INFO_LABELS:
            img = m.get(label)
            if img is not None and h_title > 0:
                h, w = img.shape[:2]
                if h != h_title:
                    new_w = int(w * h_title / h)
                    img = cv2.resize(img, (new_w, h_title))
            info_row.append(img)
        for label in DATA_LABELS:
            img = m.get(label)
            if img is not None and h_title > 0:
                h, w = img.shape[:2]
                if h != h_title:
                    new_w = int(w * h_title / h)
                    img = cv2.resize(img, (new_w, h_title))
            data_row.append(img)
        info_images.append(info_row)
        data_images.append(data_row)
    bs = 1
    bc = 22
    px = 12
    py = 12
    pc = 255
    info_canvas, info_coords_map = rander_table(
        info_images, bs, bc, px, py, pc)
    cv2.imwrite(LAYOUT_DIR/'info.png', info_canvas)
    info_theme_track = [[themes_for_rows[r] if info_images[r][c] is not None else None for c in range(
        len(INFO_LABELS))] for r in range(len(info_images))]
    store.json(LAYOUT_DIR/'info-theme-track.json', info_theme_track)
    store.json(LAYOUT_DIR/'info-pos-track.json', info_coords_map)
    data_canvas, data_coords_map = rander_table(
        data_images, bs, bc, px, py, pc)
    cv2.imwrite(LAYOUT_DIR/'data.png', data_canvas)
    data_theme_track = [[themes_for_rows[r] if data_images[r][c] is not None else None for c in range(
        len(DATA_LABELS))] for r in range(len(data_images))]
    store.json(LAYOUT_DIR/'data-theme-track.json', data_theme_track)
    store.json(LAYOUT_DIR/'data-pos-track.json', data_coords_map)
    logger.info("%s done.", TASK_NAME)
    return

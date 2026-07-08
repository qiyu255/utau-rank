from dataclasses import dataclass, field, asdict
from collections import defaultdict
from functools import lru_cache
import logging
import os
import json
import csv
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


@lru_cache(maxsize=None)
def get_boxes(name: str, width: int, height: int) -> dict[str, dict[str, int]]:
    """加载 ROI 并按图像尺寸缩放"""
    roi = None
    with open(f'data/roi_{name}.json', "r", encoding="utf-8") as f:
        roi = json.load(f)

    rw = roi["referenceWidth"]
    rh = roi["referenceHeight"]

    scale_x = width / rw
    scale_y = height / rh

    m = {}
    for box in roi["boxes"]:
        item = {}
        item["x"] = int(box["x"] * scale_x)
        item["y"] = int(box["y"] * scale_y)
        item["w"] = int(box["w"] * scale_x)
        item["h"] = int(box["h"] * scale_y)
        m[box['name']] = item
    return m


def concat(images, bg=(255, 255, 255)):
    # 计算最大高度和总宽度
    max_height = max(img.shape[0] for img in images)
    total_width = sum(img.shape[1] for img in images)

    # 创建白色背景画布
    canvas = np.full((max_height, total_width, 3), bg, dtype=np.uint8)

    # 逐个粘贴图像
    x_offset = 0
    for img in images:
        h, w = img.shape[:2]
        # 将图像粘贴到画布上
        canvas[0:h, x_offset:x_offset+w] = img
        x_offset += w

    return canvas


def crop(img, box) -> np.ndarray:
    img_h, img_w = img.shape[:2]

    x = int(box.get('x', 0))
    y = int(box.get('y', 0))
    w = int(box.get('w', img_w))
    h = int(box.get('h', img_h))

    # 越界检查：给出清晰的错误提示
    if x >= img_w or y >= img_h:
        raise ValueError(
            f"裁剪起点越界！图像尺寸为 (H={img_h}, W={img_w}, C={img.shape[2]})，"
            f"但起点 x={x}, y={y}。注意：OpenCV 的 shape 是 (H, W, C)，"
            f"你可能是把宽/高搞反了。"
        )

    # 安全钳制
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(0, min(w, img_w - x))
    h = max(0, min(h, img_h - y))

    if w == 0 or h == 0:
        raise ValueError(
            f"裁剪区域无效：x={x}, y={y}, w={w}, h={h}。 "
            f"请检查参数是否超出图像范围 (H={img_h}, W={img_w})。"
        )

    return img[y:y+h, x:x+w]


def setup_logger(
    log_file="logs/latest.log",
    logger_name="work",
):
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)      # Logger 本身最低级别
    logger.handlers.clear()             # 防止重复添加 Handler
    logger.propagate = False

    formatter = logging.Formatter(
        "%(levelname)-8s  %(message)s"
    )

    # ==========================
    # 控制台 Handler（INFO+）
    # ==========================
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # ==========================
    # 文件 Handler（DEBUG+）
    # ==========================
    file = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8"
    )
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)

    return logger


class Store:
    CSV_DIR = Path('cache/csv')
    RESULT_DIR = Path('cache/result')

    def __init__(self):
        Store.CSV_DIR.mkdir(parents=True, exist_ok=True)
        Store.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def stats(self, name, data, index, columns):
        df = pd.DataFrame(data, index, columns, copy=False)
        df.to_csv(Store.CSV_DIR / f'{name}.csv')

    def entries(self, name, data):
        data = [{
            "id": i+1,
            "best": v,
            "frames": [v],
            "indicator": {}} for i, v in enumerate(data)]
        self.gallery(name, data)

    def clusters(self, name, clus):
        items = []
        for i, c in enumerate(clus):
            item = {
                'id': i+1,
                'best': '',
                'frames': [],
                'indicator': {}
            }
            for p in c:
                item['frames'].append(Path(p).name)
            item["best"] = item['frames'][0]
            items.append(item)
        self.gallery(name, items)

    def gallery(self, name, data):
        with open(Store.RESULT_DIR / f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=lambda x: str(x))


@dataclass
class Context:
    image_paths: list = field(default_factory=list)
    output: any = field(default=None)


logger = setup_logger()
store = Store()

from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
import os
import json
import csv
import numpy as np
import cv2


from functools import lru_cache


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


logger = setup_logger()


@lru_cache(maxsize=None)
def load_roi(name: str, width: int, height: int) -> dict[str, dict[str, int | str]]:
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


def crop_image_safe(image: np.ndarray, crop_params: dict[str, int],
                    fill_color: tuple[int, int, int] | int = 0) -> np.ndarray:
    """
    安全的图像裁剪函数，当裁剪区域超出边界时用指定颜色填充

    Args:
        image: 输入图像
        crop_params: 裁剪参数
        fill_color: 超出边界时的填充颜色 (BGR格式)

    Returns:
        裁剪后的图像
    """
    x = crop_params['x']
    y = crop_params['y']
    w = crop_params['w']
    h = crop_params['h']

    img_height, img_width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    # 创建输出图像并用填充色初始化
    if len(image.shape) == 2:  # 灰度图
        output = np.full((h, w), fill_color, dtype=image.dtype)
    else:  # 彩色图
        output = np.full((h, w, channels), fill_color, dtype=image.dtype)

    # 计算有效裁剪区域
    src_x_start = max(0, x)
    src_y_start = max(0, y)
    src_x_end = min(img_width, x + w)
    src_y_end = min(img_height, y + h)

    # 计算目标区域
    dst_x_start = src_x_start - x
    dst_y_start = src_y_start - y
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    # 复制有效区域
    output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        image[src_y_start:src_y_end, src_x_start:src_x_end]

    return output


def concat_image(images, bg=(255, 255, 255)):
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


def crop_image(img, crop_params) -> np.ndarray:
    img_h, img_w = img.shape[:2]

    x = int(crop_params.get('x', 0))
    y = int(crop_params.get('y', 0))
    w = int(crop_params.get('w', img_w))
    h = int(crop_params.get('h', img_h))

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


class Sheet:
    def __init__(self, name, mode='w'):
        os.makedirs('cache/csv', exist_ok=True)
        self._f = open(f'cache/csv/{name}.csv', mode,
                       newline='', encoding='utf-8')
        self._w = csv.DictWriter(self._f, [])
        self._onec = True

    def insert(self, row: dict):
        if self._onec:
            self._w.fieldnames = row.keys()
            self._w.writeheader()
            self._onec = False

        self._w.writerow(row)

    def close(self):
        self._f.close()


# @dataclass
# class Cluster:
#     id: int = field(default=0)
#     best: str = field(default='')
#     frames: list[str] = field(default_factory=list)
#     indicator: defaultdict[str, float] = field(
#         default_factory=lambda: defaultdict(lambda:0)
#     )


# class ClusterTable:
#     def __init__(self):
#         self._data:list[Cluster] = []
#         self._seq = 0

#     def add_cluster(self,cluster:Cluster):
#         self._data.append(cluster)
#         self._seq = cluster.id

#     def add_frame(self, frame:str, indicator={}):
#         cluster = Cluster(self._seq+1)
#         cluster.best = frame
#         cluster.frames[0] = frame
#         cluster.indicator = indicator


@dataclass
class Cluster:
    id: int = field(default=0)
    best: str = field(default='')
    frames: list[str] = field(default_factory=list)
    indicator: defaultdict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    def __str__(self) -> str:
        return (f"Cluster(id={self.id}, best='{self.best}', "
                f"frames={self.frames}, indicator=dict({dict(self.indicator)}))")

    def __repr__(self) -> str:
        return (f"Cluster(id={self.id!r}, best={self.best!r}, "
                f"frames={self.frames!r}, indicator={self.indicator!r})")

    def to_dict(self) -> dict:
        """转换为普通字典，indicator 转为普通 dict"""
        d = asdict(self)
        d['indicator'] = dict(d['indicator'])
        return d

    @staticmethod
    def from_dict(data: dict) -> 'Cluster':
        # 将普通 dict 转为 defaultdict
        data['indicator'] = defaultdict(float, data.get('indicator', {}))
        return Cluster(**data)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


class ClusterTable:
    def __init__(self):
        self._data: list[Cluster] = []
        self._seq = 0

    def add_cluster(self, cluster: Cluster) -> None:
        self._data.append(cluster)
        self._seq = cluster.id

    def add_frame(self, frame: str, indicator: dict = {}) -> None:
        if indicator is None:
            indicator = {}
        cluster = Cluster(
            id=self._seq + 1,
            best=frame,
            frames=[frame],
            indicator=defaultdict(float, indicator)
        )
        self._data.append(cluster)
        self._seq = cluster.id

    def __str__(self) -> str:
        return f"ClusterTable(total_clusters={len(self._data)})"

    def __repr__(self) -> str:
        return f"ClusterTable(_data={self._data!r}, _seq={self._seq})"

    def to_list(self) -> list[dict]:
        return [c.to_dict() for c in self._data]

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_list(), ensure_ascii=False, indent=2, **kwargs)

    @staticmethod
    def from_list(clusters_data: list[dict]) -> 'ClusterTable':

        table = ClusterTable()
        for c_data in clusters_data:
            cluster = Cluster.from_dict(c_data)
            table.add_cluster(cluster)

        if table._data:
            table._seq = max(c.id for c in table._data)
        else:
            table._seq = 0
        return table

    @staticmethod
    def load(file_path):
        f = open(file_path, encoding='utf-8')
        res = json.load(f)
        f.close()
        return ClusterTable.from_list(res)
    
    def save(self, file_path):
        f = open(file_path, 'w', encoding='utf-8')
        f.write(self.to_json())
        f.close

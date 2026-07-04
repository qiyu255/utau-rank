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

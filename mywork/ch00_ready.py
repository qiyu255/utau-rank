import numpy as np
import cv2
from pathlib import Path
from common import *
TASK_NAME = Path(__file__).stem


def crush(input_path, output_path) -> list[Path]:
    image = cv2.imread(input_path)
    h, w = image.shape[:2]
    boxes = get_boxes('layout', w, h)

    simg = crop(image, boxes['stats'])
    timg = crop(image, boxes['title'])
    timg = cv2.rotate(timg, cv2.ROTATE_90_CLOCKWISE)

    crushed = concat([timg, simg])

    if output_path:
        cv2.imwrite(output_path, crushed)

    return output_path


def run(image_paths: list[Path], ctx: Context):
    N = len(image_paths)
    logger.info(f"%s start: input %d frames.", TASK_NAME, N)
    DIR = Path('cache/ready')
    DIR.mkdir(parents=True, exist_ok=True)

    output_paths = [crush(fp, DIR / fp.name) for fp in image_paths]

    logger.info(f"%s done:  output %d frames.", TASK_NAME, N)
    return output_paths

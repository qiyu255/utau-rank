import cv2
import numpy as np
import json
import os
from common import *

FRAME_DIR = 'cache/crush-frames'
INPUT_PATH = "cache/01_sift_hit.json"
OUTPUT_PATH = "cache/02_cluster.json"
SHEET_ID = '02_cluster'


def example():
    details:dict[str, float|int] = {}
    details['theme']
    frame_name = ''
    #记录frame每个分类依据和相关指标
    submit_sheet(SHEET_ID, {'ts':frame_name, **details})
    logger.debug(f'{frame_name} {details}')

def pipeline(files):
    for file_name in files:
        file_path = os.path.join(FRAME_DIR, file_name)
        frame_name = os.path.splitext(file_name)[0]
        im = cv2.imread()
        h, w = im.shape[:2]

        boxes = load_roi('line', w, h)
        for lebal, box in boxes.items():
            roi = crop_image(im, boxes) # 返回裁剪后的图像






def main():
    logger.info(f"theme cluster start: load {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    files = [i['best'] for i in data]
    logger.info(f"  {len(files)} frames.")

    result = pipeline(files)


    flush_submit()
    logger.info(f"theme cluster done. saved {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

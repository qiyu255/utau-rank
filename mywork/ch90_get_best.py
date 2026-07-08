import cv2
import numpy as np
from pathlib import Path
from itertools import chain
from common import *
TASK_NAME = Path(__file__).stem
CLOUMNS = ['score', 'blur', 'bright', 'contrast', 'entropy', 'dark', 'overexp']


def analyze_image(rst, img):
    mean_v = (np.mean(img))
    std_v = (np.std(img))
    blur_v = (cv2.Laplacian(img, cv2.CV_64F).var())
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    ent_v = (-np.sum(hist * np.log2(hist)))
    dark_v = (np.mean(img < 30))
    over_v = (np.mean(img > 225))
    score = blur_v * 0.5 + std_v * 0.3 + ent_v * 0.2
    score = score / (1.0 + dark_v * 10.0 + over_v * 10.0)
    rst[0] = score
    rst[1] = blur_v
    rst[2] = mean_v
    rst[3] = std_v
    rst[4] = ent_v
    rst[5] = dark_v
    rst[6] = over_v


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def analyze_cluster(rst, image_paths) -> np.ndarray:
    for i in range(len(image_paths)):
        img = cv2.imread(str(image_paths[i]))
        analyze_image(rst[i], preprocess(img))


def run(clusters: list[list[Path]], ctx: Context):
    N = len(clusters)
    logger.info(f"%s start: input %d clusters.", TASK_NAME, N)
    ids = [i.stem for i in chain.from_iterable(clusters)]
    stats = np.empty((len(ids), len(CLOUMNS)), dtype=np.float64)
    bests = []
    row_pos = 0
    for c in clusters:
        cs = stats[row_pos:row_pos + len(c)]
        analyze_cluster(cs, c)
        row_pos += len(c)
        x = np.argmax(cs[:, 0])
        bests.append(c[x])
    store.stats(TASK_NAME, stats, ids, CLOUMNS)
    store.bests(TASK_NAME, bests, clusters)
    logger.info(f"%s done:  output %d items.", TASK_NAME, len(bests))
    return bests

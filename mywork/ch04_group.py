import cv2
import numpy as np
from itertools import compress
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from common import *
TASK_NAME = Path(__file__).stem
FEAT_NAMES = ['mean_a', 'mean_b', 'std_a', 'std_b']
N_CLUSTERS = 5
N_COMPONENTS = 4


def bg_mask(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] > 20) | (hsv[:, :, 2] < 245)


def roi_feat(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(4)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1][mask]
    b = lab[:, :, 2][mask]
    return np.array([np.mean(a), np.mean(b), np.std(a), np.std(b)])


def scan(image_paths: list[Path]) -> tuple:
    n = len(image_paths)
    feats = np.empty((n, len(FEAT_NAMES)), dtype=np.float64)
    colors = np.empty((n, 3), dtype=np.float64)
    for i, path in enumerate(image_paths):
        im = cv2.imread(str(path))
        boxes = get_boxes('line', im.shape[1], im.shape[0])
        rois = [crop(im, box) for box in boxes.values()]
        masks = [bg_mask(r) for r in rois]
        feats[i] = np.mean([roi_feat(r, m)
                           for r, m in zip(rois, masks)], axis=0)
        fg = [np.mean(r[m], axis=0) for r, m in zip(rois, masks) if m.any()]
        colors[i] = np.mean(fg, axis=0) if fg else np.array(
            [128.0, 128.0, 128.0])
    return feats, colors


def run(image_paths: list[Path], ctx: Context):
    n = len(image_paths)
    logger.info(f"%s start: input %d frames.", TASK_NAME, n)
    feats, colors = scan(image_paths)
    store.stats(TASK_NAME, feats, [p.stem for p in image_paths], FEAT_NAMES)
    nc = min(N_COMPONENTS, n, feats.shape[1])
    pca = PCA(n_components=nc)
    x = pca.fit_transform(feats)
    k = min(N_CLUSTERS, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(x)
    groups = []
    hsv_list = []
    for label in sorted(set(labels)):
        idx = labels == label
        group = list(compress(image_paths, idx))
        bgr = colors[idx].mean(axis=0).astype(np.uint8)
        hsv = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        store.entries(f"{TASK_NAME}_c{label}", [p.name for p in group])
        groups.append(group)
        hsv_list.append(hsv.tolist())
    logger.info(f"%s done: output %d groups.", TASK_NAME, k)
    return groups, hsv_list

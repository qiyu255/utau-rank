import cv2
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from common import *

FRAME_DIR = 'cache/crush-frames'
INPUT_PATH = "cache/01_sift_hit.json"
OUTPUT_PATH = "cache/02_cluster.json"
SHEET_ID = '02_cluster'


def bg_mask(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    return (s > 20) | (v < 245)


def lab_stats(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1][mask]
    b = lab[:, :, 2][mask]
    return np.array([np.mean(a), np.mean(b), np.std(a), np.std(b)])


def ab_hist(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [1, 2], mask.astype(
        np.uint8), [16, 16], [0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten()
    return hist


def roi_feat(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(260)
    stats = lab_stats(roi, mask)
    hist = ab_hist(roi, mask)
    return np.concatenate([stats, hist])


def frame_feat(im: np.ndarray, boxes: dict) -> np.ndarray:
    feats = []
    for box in boxes.values():
        roi = crop_image(im, box)
        mask = bg_mask(roi)
        feats.append(roi_feat(roi, mask))
    return np.mean(feats, axis=0)


def frame_color(im: np.ndarray, boxes: dict) -> np.ndarray:
    colors = []
    for box in boxes.values():
        roi = crop_image(im, box)
        mask = bg_mask(roi)
        if mask.any():
            colors.append(np.mean(roi[mask], axis=0))
    return np.mean(colors, axis=0) if colors else np.array([128, 128, 128])


def fit_model(X: np.ndarray) -> tuple[PCA, KMeans]:
    n = min(8, X.shape[0], X.shape[1])
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    k = min(5, X.shape[0])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    return pca, km


def theme_palette(labels: np.ndarray, colors: list[np.ndarray]) -> list[dict]:
    palette = []
    for theme in sorted(set(labels)):
        theme_colors = [colors[i] for i, l in enumerate(labels) if l == theme]
        avg = np.mean(theme_colors, axis=0).astype(int).tolist()
        palette.append({'theme': int(theme), 'bgr': avg})
    return palette


def pipeline(files: list[str]) -> dict:
    paths = [os.path.join(FRAME_DIR, f) for f in files]

    feats = []
    colors = []
    for p in paths:
        im = cv2.imread(p)
        h, w = im.shape[:2]
        boxes = load_roi('line', w, h)
        feats.append(frame_feat(im, boxes))
        colors.append(frame_color(im, boxes))

    X = np.array(feats)
    pca, km = fit_model(X)
    X_pca = pca.transform(X)
    labels = km.predict(X_pca)

    palette = theme_palette(labels, colors)

    result = {'palette': palette, 'frames': []}
    for i, file in enumerate(files):
        name = os.path.splitext(file)[0]

        feat = feats[i]
        details = {
            'theme': int(labels[i]),
            'mean_a': round(float(feat[0]), 2),
            'mean_b': round(float(feat[1]), 2),
            'std_a': round(float(feat[2]), 2),
            'std_b': round(float(feat[3]), 2),
            'pca_var': round(float(np.sum(pca.explained_variance_ratio_)), 4),
        }
        submit_sheet(SHEET_ID, {'ts': name, **details})
        result['frames'].append({'file': file, 'details': details})
        logger.debug(f'{file} {details}')

    return result


def main():
    logger.info(f"theme cluster start: load {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    files = [i['best'] for i in data]
    # logger.info(f"  {len(files)} frames.")

    result = pipeline(files)
    palette = result['palette']
    clusters = defaultdict(Cluster)
    for frame in result['frames']:
        name = frame['file']
        t = frame['details']['theme']
        clusters[t].id = t
        clusters[t].best = name
        clusters[t].frames.append(name)
        clusters[t].indicator = {'color': palette[t],  **frame['details']}

    ct = ClusterTable(OUTPUT_PATH)
    for c in clusters.values():
        ct.add_cluster(c)
    ct.save()

    flush_submit()
    logger.info(f"theme cluster done. saved {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

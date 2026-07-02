import cv2
import numpy as np
import json
import glob
import os

# =========================
# 全局参数（可调）
# =========================
FRAME_DIR = "cache/frames/*.png"
ROI_PATH = "data/roi_layout.json"
OUTPUT_PATH = "cache/00_filter.json"

THRESHOLD_LOW = 180
THRESHOLD_HIGH = 250

LAPLACIAN_LOW_QUALITY = 30  # 可调：越低越糊

# =========================
# ROI 读取与缩放
# =========================
def load_roi_layout(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def scale_roi(box, img_w, img_h, ref_w, ref_h):
    return {
        "x": int(box["x"] * img_w / ref_w),
        "y": int(box["y"] * img_h / ref_h),
        "w": int(box["w"] * img_w / ref_w),
        "h": int(box["h"] * img_h / ref_h),
    }

def extract_rois(img, layout):
    h, w = img.shape[:2]
    ref_w = layout["referenceWidth"]
    ref_h = layout["referenceHeight"]

    rois = {}
    for b in layout["boxes"]:
        if b["name"] not in ("stats", "title"):
            continue
        r = scale_roi(b, w, h, ref_w, ref_h)
        crop = img[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]
        rois[b["name"]] = crop
    return rois

# =========================
# 亮度计算
# =========================
def calc_brightness(gray_img):
    return float(np.mean(gray_img))

def frame_brightness(img, rois):
    vals = []
    for _, roi_img in rois.items():
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        vals.append(calc_brightness(gray))
    return float(np.mean(vals)) if vals else 0.0

# =========================
# 清晰度（Laplacian）
# =========================
def laplacian_var(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def is_low_quality(lap_var):
    return lap_var < LAPLACIAN_LOW_QUALITY

# =========================
# 分类逻辑
# =========================
def classify(brightness):
    if THRESHOLD_LOW <= brightness <= THRESHOLD_HIGH:
        return "retain"
    return "abandon"

# =========================
# cluster 管理
# =========================
def get_cluster(store, step):
    if step not in store:
        store[step] = {
            "cluster_id": len(store),
            "step": step,
            "max_laplacian_var": 0.0,
            "low_quality": False,
            "representative_frame": None,
            "frames": []
        }
    return store[step]

def update_cluster(cluster, frame_name, lap_var, low_q):
    if not cluster["frames"]:
        cluster["representative_frame"] = frame_name

    cluster["frames"].append(frame_name)
    cluster["max_laplacian_var"] = max(cluster["max_laplacian_var"], lap_var)
    cluster["low_quality"] = cluster["low_quality"] or low_q

# =========================
# 主流程（流式）
# =========================
def process():
    layout = load_roi_layout(ROI_PATH)
    frames = sorted(glob.glob(FRAME_DIR))

    clusters = {}

    for path in frames:
        img = cv2.imread(path)
        if img is None:
            continue

        name = os.path.basename(path)

        rois = extract_rois(img, layout)

        brightness = frame_brightness(img, rois)
        lap_var = laplacian_var(img)
        low_q = is_low_quality(lap_var)

        step = classify(brightness)

        cluster = get_cluster(clusters, step)
        update_cluster(cluster, name, lap_var, low_q)

    # 按 cluster_id 排序输出
    result = sorted(clusters.values(), key=lambda x: x["cluster_id"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"saved -> {OUTPUT_PATH}")

# =========================
# 入口
# =========================
if __name__ == "__main__":
    process()
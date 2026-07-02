import cv2
import numpy as np
import json
import os
import shutil
from glob import glob

# ================== 全局配置参数 ==================
ROI_LAYOUT_PATH = "data/roi_layout.json"
FRAMES_DIR = "cache/frames"
OUTPUT_DIR = "cache/purified"
OUTPUT_JSON = "cache/02_purified.json"

# 感知哈希差异阈值 (5% 对应汉明距离 ≤ 3，针对 64 位 pHash)
HASH_THRESHOLD = 2

# 拉普拉斯方差低于此值视为“低质量簇”（爆浆页）
LAPLACIAN_LOW_QUALITY_THRESHOLD = 100

# 淡入淡出及极暗/极亮帧过滤参数
LUM_LOW_THRESHOLD = 20       # 极暗帧丢弃阈值 (stats 区域平均灰度值)
LUM_HIGH_THRESHOLD = 235     # 极亮帧丢弃阈值 (stats 区域平均灰度值)
GRADIENT_MIN_LENGTH = 3      # 渐变最少连续帧数
GRADIENT_DELTA = 5.0         # 渐变相邻帧亮度差阈值
GRADIENT_MARGIN = 2          # 渐变区间扩展边距

# 光流对齐参数
FLOW_MEDIAN_BLUR = True      # 合成时对时序取中值；False 则为均值

# 文件扫描通配符
FRAME_PATTERN = "*.png"
# =================================================

# 检查 cv2.img_hash 是否可用
if not hasattr(cv2, 'img_hash'):
    raise RuntimeError("当前 OpenCV 版本不支持 img_hash 模块，请安装 opencv-contrib-python")

def load_roi_layout(path):
    with open(path, "r") as f:
        layout = json.load(f)
    return layout

def adapt_roi(box, img_w, img_h, ref_w, ref_h):
    scale_x = img_w / ref_w
    scale_y = img_h / ref_h
    x = int(box["x"] * scale_x)
    y = int(box["y"] * scale_y)
    w = int(box["w"] * scale_x)
    h = int(box["h"] * scale_y)
    return x, y, w, h

def compute_brightness(img, roi):
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return 0.0
    roi_img = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def filter_fade_and_extreme_frames(brightness_list,
                                   low_thresh=LUM_LOW_THRESHOLD,
                                   high_thresh=LUM_HIGH_THRESHOLD,
                                   min_gradient_len=GRADIENT_MIN_LENGTH,
                                   grad_delta=GRADIENT_DELTA,
                                   margin=GRADIENT_MARGIN):
    """
    返回需要保留的布尔掩码 (list of bool)
    - 丢弃亮度低于 low_thresh 或高于 high_thresh 的帧（极暗/极亮）
    - 丢弃所有检测到的单调渐变区间内的帧（淡入/淡出）
    """
    n = len(brightness_list)
    keep = [True] * n

    # 1. 标记极暗/极亮帧
    for i in range(n):
        val = brightness_list[i]
        if val < low_thresh or val > high_thresh:
            keep[i] = False

    # 2. 检测渐变区间（单调递增/递减，变化量足够）
    if n < min_gradient_len + 1:
        return keep

    diffs = np.diff(brightness_list)
    i = 0
    while i < n - 1:
        if abs(diffs[i]) <= grad_delta:
            i += 1
            continue
        direction = np.sign(diffs[i])
        start = i
        while i < n - 1 and np.sign(diffs[i]) == direction and abs(diffs[i]) > grad_delta:
            i += 1
        end = i  # end 可能为 n-1（此时差分已越界），实际帧对应 end+1
        if end >= n - 1:
            end = n - 2  # 限制为最后一个有效差分索引

        # 计算实际渐变包含的帧数：从 start 到 end+1
        length = end + 1 - start + 1  # = end - start + 2
        if length >= min_gradient_len:
            total_change = abs(brightness_list[end+1] - brightness_list[start])
            if total_change > grad_delta * length:
                # 标记渐变帧区域，并向外扩展 margin
                fade_start = max(0, start - margin)
                fade_end = min(n - 1, end + 1 + margin)
                for j in range(fade_start, fade_end + 1):
                    keep[j] = False
        # 继续从 end+1 开始扫描下一段
        i = end + 1
    return keep

def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def blacken_cover(image, cover_roi):
    x, y, w, h = cover_roi
    if w > 0 and h > 0:
        image[y:y+h, x:x+w] = 0
    return image

def restore_cover(synthetic, original, cover_roi):
    x, y, w, h = cover_roi
    if w > 0 and h > 0:
        synthetic[y:y+h, x:x+w] = original[y:y+h, x:x+w]
    return synthetic

def align_and_fuse(frames_path, ref_path, cover_roi):
    ref = cv2.imread(ref_path)
    if ref is None:
        raise ValueError(f"无法读取参考帧: {ref_path}")
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    try:
        optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    except AttributeError:
        try:
            optical_flow = cv2.DualTVL1OpticalFlow_create()
        except AttributeError:
            raise RuntimeError("DualTVL1 光流不可用，请安装 opencv-contrib-python")

    warped_frames = []
    for fpath in frames_path:
        img = cv2.imread(fpath)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = optical_flow.calc(gray, ref_gray, None)
        h, w = flow.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + flow[..., 0]).astype(np.float32)
        map_y = (map_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        warped_frames.append(warped)

    if not warped_frames:
        return ref

    stack = np.stack(warped_frames, axis=0)
    if FLOW_MEDIAN_BLUR:
        fused = np.median(stack, axis=0).astype(np.uint8)
    else:
        fused = np.mean(stack, axis=0).astype(np.uint8)

    fused = restore_cover(fused, ref, cover_roi)
    return fused

def process():
    layout = load_roi_layout(ROI_LAYOUT_PATH)
    ref_w = layout["referenceWidth"]
    ref_h = layout["referenceHeight"]
    boxes = {b["name"]: b for b in layout["boxes"]}
    if "stats" not in boxes or "cover" not in boxes:
        raise ValueError("ROI 布局中必须包含 'stats' 和 'cover'")

    frame_files = sorted(glob(os.path.join(FRAMES_DIR, FRAME_PATTERN)))
    if not frame_files:
        print("没有找到任何帧文件，退出。")
        return

    # 第一遍：读取亮度序列
    print("第一遍：读取亮度序列...")
    brightness = []
    first_img = cv2.imread(frame_files[0])
    if first_img is None:
        raise RuntimeError(f"无法读取第一帧: {frame_files[0]}")
    img_h, img_w = first_img.shape[:2]
    stats_roi = adapt_roi(boxes["stats"], img_w, img_h, ref_w, ref_h)
    cover_roi = adapt_roi(boxes["cover"], img_w, img_h, ref_w, ref_h)

    brightness.append(compute_brightness(first_img, stats_roi))
    for fpath in frame_files[1:]:
        img = cv2.imread(fpath)
        if img is None:
            continue
        brightness.append(compute_brightness(img, stats_roi))

    # 过滤淡入淡出及极暗/极亮帧
    keep_mask = filter_fade_and_extreme_frames(brightness)
    retained_files = [f for f, k in zip(frame_files, keep_mask) if k]
    print(f"原始帧数: {len(frame_files)}, 保留帧数: {len(retained_files)}")

    if len(retained_files) == 0:
        print("所有帧均被过滤，退出。")
        return

    # 第二遍：计算感知哈希（pHashes）和拉普拉斯方差
    print("第二遍：计算哈希与清晰度...")
    hasher = cv2.img_hash.PHash_create()

    hashes = []
    laplacian_vars = []
    for fpath in retained_files:
        img = cv2.imread(fpath)
        if img is None:
            hashes.append(None)
            laplacian_vars.append(0.0)
            continue
        # 黑化 cover 区域，避免封面内容干扰哈希
        img_hashed = img.copy()
        blacken_cover(img_hashed, cover_roi)
        hash_val = hasher.compute(img_hashed)
        hashes.append(hash_val)
        var = laplacian_variance(img)
        laplacian_vars.append(var)

    # 聚类：使用 cv2.img_hash 的 compare 方法
    print("聚类页面簇...")
    clusters = []
    current_cluster = [0]
    hash_threshold = HASH_THRESHOLD
    for i in range(1, len(hashes)):
        if hashes[i-1] is None or hashes[i] is None:
            current_cluster.append(i)
            continue
        dist = hasher.compare(hashes[i-1], hashes[i])
        if dist <= hash_threshold:
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]
    clusters.append(current_cluster)

    print(f"共形成 {len(clusters)} 个页面簇")

    # 处理每个簇
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_info = []
    for cluster_id, indices in enumerate(clusters):
        print(f"处理簇 {cluster_id}，包含 {len(indices)} 帧")
        cluster_vars = [laplacian_vars[idx] for idx in indices]
        max_var = max(cluster_vars)
        best_idx_in_cluster = indices[cluster_vars.index(max_var)]
        rep_path = retained_files[best_idx_in_cluster]

        is_low_quality = max_var < LAPLACIAN_LOW_QUALITY_THRESHOLD
        output_name = os.path.basename(rep_path)
        output_path = os.path.join(OUTPUT_DIR, output_name)

        if is_low_quality:
            print(f"  低质量簇，最大方差 {max_var:.2f} < {LAPLACIAN_LOW_QUALITY_THRESHOLD}，进行光流对齐融合...")
            cluster_paths = [retained_files[idx] for idx in indices]
            try:
                fused_img = align_and_fuse(cluster_paths, rep_path, cover_roi)
                cv2.imwrite(output_path, fused_img)
                print(f"  合成图已保存: {output_path}")
            except Exception as e:
                print(f"  光流融合失败: {e}，回退为普通代表帧")
                shutil.copy2(rep_path, output_path)
        else:
            shutil.copy2(rep_path, output_path)
            print(f"  代表帧已复制: {output_path}")

        cluster_info = {
            "cluster_id": cluster_id,
            # "num_frames": len(indices),
            "max_laplacian_var": float(max_var),          # 转为原生 Python float
            "low_quality": bool(is_low_quality),          # 转为原生 Python bool
            "representative_frame": output_name,
            "frames": [os.path.basename(retained_files[idx]) for idx in indices]
        }
        result_info.append(cluster_info)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result_info, f, indent=2, ensure_ascii=False)
    print(f"关联信息已保存至 {OUTPUT_JSON}")
    print("处理完成！")

if __name__ == "__main__":
    process()
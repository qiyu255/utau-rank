import os
import cv2
import json
import glob
import shutil
import numpy as np
import imagehash
from PIL import Image
# ==========================================
# 全局配置参数
# ==========================================
CONFIG = {
    "roi_layout_path": "data/roi_layout.json",
    "frames_dir": "cache/frames",
    "output_dir": "cache/purified",
    "output_json": "cache/02_purified.json",
    # 感知哈希差异阈值 (5% 对应 64位哈希的汉明距离约为 3.2)
    "phash_hamming_threshold": 4,  
    # 拉普拉斯方差全局阈值，低于此值判定为"爆浆页/低质量簇"
    "blur_threshold": 100.0,
    # 亮度过滤阈值，用于丢弃淡入淡出过渡帧
    "brightness_min": 15.0,
    "brightness_max": 240.0,
}
def get_cover_box(roi_data):
    """从 roi_layout.json 提取 cover 区域配置"""
    for box in roi_data.get("boxes", []):
        if box.get("name") == "cover":
            return box
    return None
def apply_cover_mask(img, cover_box):
    """将图像的 cover 区域填充为黑色以排除干扰"""
    masked = img.copy()
    if cover_box:
        x, y, w, h = cover_box['x'], cover_box['y'], cover_box['w'], cover_box['h']
        masked[y:y+h, x:x+w] = (0, 0, 0)
    return masked
def calc_optical_flow_align(rep_gray, curr_gray):
    """使用 DualTVL1 计算光流"""
    try:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    except AttributeError:
        tvl1 = cv2.optflow.createOptFlow_DualTVL1()
    flow = tvl1.calc(rep_gray, curr_gray, None)
    return flow
def align_and_merge_frames(cluster, rep_idx):
    """
    使用光流将簇内所有帧对齐到代表帧，并取中值合并以抹除噪点和马赛克
    """
    rep_item = cluster[rep_idx]
    rep_img = rep_item["img"]
    rep_gray = cv2.cvtColor(rep_img, cv2.COLOR_BGR2GRAY)
    h, w = rep_gray.shape
    map_x_base, map_y_base = np.meshgrid(np.arange(w), np.arange(h))
    map_x_base = map_x_base.astype(np.float32)
    map_y_base = map_y_base.astype(np.float32)
    aligned_imgs = [rep_img]
    for i, item in enumerate(cluster):
        if i == rep_idx:
            continue
        curr_img = item["img"]
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        # 计算当前帧到代表帧的光流
        flow = calc_optical_flow_align(rep_gray, curr_gray)
        # 根据光流生成映射矩阵并扭曲图像
        map_x = map_x_base + flow[:, :, 0]
        map_y = map_y_base + flow[:, :, 1]
        warped = cv2.remap(curr_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aligned_imgs.append(warped)
    # 像素级取中值
    merged = np.median(np.array(aligned_imgs), axis=0).astype(np.uint8)
    return merged
def process_cluster(cluster, cluster_id, cover_box, output_data):
    """处理单个页面簇：提取代表帧或执行去噪合成"""
    if not cluster:
        return
    # 找出拉普拉斯方差最大的帧作为初始代表帧
    best_idx = max(range(len(cluster)), key=lambda i: cluster[i]["blur_var"])
    rep_item = cluster[best_idx]
    max_var = rep_item["blur_var"]
    is_synthesized = False
    if max_var < CONFIG["blur_threshold"]:
        # 整簇低质量（爆浆页），触发光流对齐与中值合成
        merged_img = align_and_merge_frames(cluster, best_idx)
        is_synthesized = True
        # 恢复 cover 区域（从代表帧原图中截取贴回）
        if cover_box:
            x, y, w, h = cover_box['x'], cover_box['y'], cover_box['w'], cover_box['h']
            merged_img[y:y+h, x:x+w] = rep_item["img"][y:y+h, x:x+w]
        out_filename = os.path.basename(rep_item["file"])
        out_path = os.path.join(CONFIG["output_dir"], out_filename)
        cv2.imwrite(out_path, merged_img)
    else:
        # 高质量簇，优先使用文件复制直接保留原图
        out_filename = os.path.basename(rep_item["file"])
        out_path = os.path.join(CONFIG["output_dir"], out_filename)
        shutil.copyfile(rep_item["file"], out_path)
    output_data.append({
        "cluster_id": cluster_id,
        "frames": [os.path.basename(item["file"]) for item in cluster],
        "representative_frame": out_filename,
        "is_synthesized": is_synthesized,
        "max_blur_variance": float(max_var)
    })
def main():
    # 初始化环境
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(CONFIG["roi_layout_path"], 'r', encoding='utf-8') as f:
        roi_data = json.load(f)
    cover_box = get_cover_box(roi_data)
    # 流式读取帧文件并排序
    files = sorted(glob.glob(os.path.join(CONFIG["frames_dir"], "*.png")))
    current_cluster = []
    prev_phash = None
    cluster_id = 0
    output_data = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        # 屏蔽 cover 区域以进行后续无干扰的特征计算
        masked_img = apply_cover_mask(img, cover_box)
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        # 1. 丢弃亮度淡入淡出的过渡帧
        brightness = np.mean(gray)
        if brightness < CONFIG["brightness_min"] or brightness > CONFIG["brightness_max"]:
            continue
        # 2. 计算拉普拉斯方差（模糊度）和 pHash
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        curr_phash = imagehash.phash(Image.fromarray(gray))
        # 3. 根据 pHash 差异判断是否归属于同一簇
        if prev_phash is not None:
            hamming_dist = curr_phash - prev_phash
            if hamming_dist > CONFIG["phash_hamming_threshold"]:
                # 差异过大，归属不同页面。处理并清空当前簇。
                process_cluster(current_cluster, cluster_id, cover_box, output_data)
                cluster_id += 1
                current_cluster = []
        current_cluster.append({
            "file": f,
            "img": img,         # 保留原图用于合成及恢复cover
            "blur_var": blur_var,
            "phash": curr_phash
        })
        prev_phash = curr_phash
    # 处理最后一个簇
    process_cluster(current_cluster, cluster_id, cover_box, output_data)
    # 输出关联信息 JSON
    with open(CONFIG["output_json"], 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
if __name__ == "__main__":
    main()
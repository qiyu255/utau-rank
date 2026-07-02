import os
import json
import logging
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Generator
# ==================== 全局参数 ====================
TITLE_WEIGHT: float = 0.5
STATS_WEIGHT: float = 0.5
GAUSSIAN_KERNEL: int = 5
PIXEL_DIFF_THRESHOLD: int = 30
MORPH_KERNEL: int = 3
SSIM_THRESHOLD: float = 0.85
DIFF_THRESHOLD: float = 0.05
FILTER_PATH: str = "cache/00_filter.json"
FRAMES_DIR: str = "cache/frames/"
ROI_PATH: str = "data/roi_layout.json"
OUTPUT_PATH: str = "cache/01_cluster.json"
# ==================== 日志配置 ====================
def setup_logging() -> None:
    """配置日志格式与级别"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
# ==================== 数据加载 ====================
def load_json(path: str) -> Any:
    """安全加载 JSON 文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def load_filter(path: str) -> List[Dict[str, Any]]:
    """加载过滤数据"""
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Filter JSON 格式错误: 期望列表")
    return data
def load_roi_layout(path: str) -> Dict[str, Any]:
    """加载 ROI 布局"""
    data = load_json(path)
    if "boxes" not in data or "referenceWidth" not in data or "referenceHeight" not in data:
        raise ValueError("ROI JSON 缺少必要字段 (boxes, referenceWidth, referenceHeight)")
    return data
def iter_frames(filter_data: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """流式迭代所有帧名，避免一次性加载"""
    # for cluster in filter_data:
    for frame in filter_data[0].get("frames", []):
        yield frame
# ==================== 图像与 ROI 处理 ====================
def read_gray(path: str) -> np.ndarray:
    """读取图像并转为灰度图"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"图片不存在: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"灰度转换失败或图片损坏: {path}")
    return img
def scale_roi(roi: Dict[str, Any], ref_w: int, ref_h: int, img_w: int, img_h: int) -> Dict[str, int]:
    """按比例缩放 ROI 并裁剪边界"""
    sx = img_w / ref_w
    sy = img_h / ref_h
    x = int(roi["x"] * sx)
    y = int(roi["y"] * sy)
    w = int(roi["w"] * sx)
    h = int(roi["h"] * sy)
    x = max(0, x)
    y = max(0, y)
    w = min(img_w - x, w)
    h = min(img_h - y, h)
    return {"x": x, "y": y, "w": w, "h": h}
def validate_roi_bounds(roi: Dict[str, int], img_w: int, img_h: int) -> None:
    """验证 ROI 是否越界"""
    if roi["w"] <= 0 or roi["h"] <= 0:
        raise ValueError(f"ROI 尺寸无效: {roi}")
    if roi["x"] + roi["w"] > img_w or roi["y"] + roi["h"] > img_h:
        raise ValueError(f"ROI 超出图片边界: {roi}, 图片尺寸: ({img_w}, {img_h})")
def init_rois(roi_layout: Dict[str, Any], img_w: int, img_h: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    """初始化并缩放所需的 title 和 stats ROI"""
    ref_w = roi_layout["referenceWidth"]
    ref_h = roi_layout["referenceHeight"]
    roi_title_orig = next((b for b in roi_layout["boxes"] if b["name"] == "title"), None)
    roi_stats_orig = next((b for b in roi_layout["boxes"] if b["name"] == "stats"), None)
    if not roi_title_orig or not roi_stats_orig:
        raise ValueError("ROI 布局中缺少 title 或 stats 区域")
    roi_title = scale_roi(roi_title_orig, ref_w, ref_h, img_w, img_h)
    roi_stats = scale_roi(roi_stats_orig, ref_w, ref_h, img_w, img_h)
    validate_roi_bounds(roi_title, img_w, img_h)
    validate_roi_bounds(roi_stats, img_w, img_h)
    return roi_title, roi_stats
def extract_roi(img: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    """从图像中截取 ROI 区域"""
    return img[roi["y"]:roi["y"]+roi["h"], roi["x"]:roi["x"]+roi["w"]]
# ==================== 差异与相似度计算 ====================
def compute_difference(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算基于高斯模糊、绝对差分、阈值化与形态学操作的变化像素比例"""
    blur1 = cv2.GaussianBlur(img1, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    blur2 = cv2.GaussianBlur(img2, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    diff = cv2.absdiff(blur1, blur2)
    _, thresh = cv2.threshold(diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    changed_pixels = np.count_nonzero(opened)
    total_pixels = opened.size
    return float(changed_pixels / total_pixels) if total_pixels > 0 else 0.0
def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """手动实现 SSIM 结构相似性指数"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))
def compare_frame(prev_title: np.ndarray, curr_title: np.ndarray, 
                    prev_stats: np.ndarray, curr_stats: np.ndarray) -> Tuple[float, float]:
    """综合计算两帧之间的加权差异与加权 SSIM"""
    title_diff = compute_difference(prev_title, curr_title)
    stats_diff = compute_difference(prev_stats, curr_stats)
    weighted_diff = TITLE_WEIGHT * title_diff + STATS_WEIGHT * stats_diff
    title_ssim = compute_ssim(prev_title, curr_title)
    stats_ssim = compute_ssim(prev_stats, curr_stats)
    weighted_ssim = TITLE_WEIGHT * title_ssim + STATS_WEIGHT * stats_ssim
    return weighted_diff, weighted_ssim
# ==================== 聚类管理 ====================
def start_cluster(frame_name: str, cluster_id: int) -> Dict[str, Any]:
    """开始一个新的页面簇"""
    return {
        "cluster_id": cluster_id,
        "step": "cluster",
        "max_laplacian_var": 0,
        "low_quality": False,
        "representative_frame": frame_name,
        "frames": [frame_name]
    }
def finish_cluster(cluster: Dict[str, Any], output_list: List[Dict[str, Any]]) -> None:
    """结束当前页面簇并存入列表"""
    output_list.append(cluster)
def save_clusters(clusters: List[Dict[str, Any]], path: str) -> None:
    """保存聚类结果到 JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
# ==================== 主程序 ====================
def main() -> None:
    """主控流式处理逻辑"""
    setup_logging()
    try:
        filter_data = load_filter(FILTER_PATH)
        roi_layout = load_roi_layout(ROI_PATH)
    except Exception as e:
        logging.error(f"初始化加载失败: {e}")
        return
    clusters: List[Dict[str, Any]] = []
    current_cluster: Dict[str, Any] = None
    cluster_id = 0
    prev_title: np.ndarray = None
    prev_stats: np.ndarray = None
    roi_title_scaled: Dict[str, int] = None
    roi_stats_scaled: Dict[str, int] = None
    for frame_name in iter_frames(filter_data):
        
        img_path = os.path.join(FRAMES_DIR, frame_name)
        try:
            gray = read_gray(img_path)
        except Exception as e:
            logging.error(f"图片读取异常: {e}")
            raise
        img_h, img_w = gray.shape
        if roi_title_scaled is None:
            try:
                roi_title_scaled, roi_stats_scaled = init_rois(roi_layout, img_w, img_h)
            except Exception as e:
                logging.error(f"ROI 初始化失败: {e}")
                raise
        curr_title = extract_roi(gray, roi_title_scaled)
        curr_stats = extract_roi(gray, roi_stats_scaled)
        if current_cluster is None:
            current_cluster = start_cluster(frame_name, cluster_id)
            cluster_id += 1
            logging.info(f"建立新簇 {current_cluster['cluster_id']}, 代表帧: {frame_name}")
        else:
            w_diff, w_ssim = compare_frame(prev_title, curr_title, prev_stats, curr_stats)
            is_same = (w_ssim > SSIM_THRESHOLD) and (w_diff < DIFF_THRESHOLD)
            logging.info(f"簇 {current_cluster['cluster_id']} - 帧: {frame_name} | w_diff: {w_diff:.4f}, w_ssim: {w_ssim:.4f} | 判定: {'同页' if is_same else '新页'}")
            if is_same:
                current_cluster["frames"].append(frame_name)
            else:
                finish_cluster(current_cluster, clusters)
                current_cluster = start_cluster(frame_name, cluster_id)
                cluster_id += 1
                logging.info(f"建立新簇 {current_cluster['cluster_id']}, 代表帧: {frame_name}")
        prev_title = curr_title
        prev_stats = curr_stats
    if current_cluster is not None:
        finish_cluster(current_cluster, clusters)
    try:
        save_clusters(clusters, OUTPUT_PATH)
        logging.info(f"聚类完成，共 {len(clusters)} 个簇，已保存至 {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"结果保存失败: {e}")
if __name__ == "__main__":
    main()
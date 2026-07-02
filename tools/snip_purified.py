import sys
import re
import os
from typing import List, Optional

RANGE = 50

def remove_timestamp(line: str) -> str:
    pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - '
    return re.sub(pattern, '', line)

def extract_frame_number(line: str) -> Optional[int]:
    match = re.search(r'_(\d+)\.png', line)
    if match:
        return int(match.group(1))
    return None

def write_section(out_lines: List[str],
                  header: Optional[str],
                  section_lines: List[str],
                  target: int,
                  min_id: int,
                  max_id: int) -> bool:
    """
    处理三个部分（亮度序列、过滤原因、哈希值），按帧编号范围裁剪。
    返回是否输出了内容。
    """
    if not section_lines:
        return False

    filtered_with_idx = []
    for idx, line in enumerate(section_lines):
        num = extract_frame_number(line)
        if num is not None and min_id <= num <= max_id:
            filtered_with_idx.append((idx, line))

    if not filtered_with_idx:
        return False  # 无内容，跳过标题

    if header is not None:
        out_lines.append(header)

    first_idx = filtered_with_idx[0][0]
    last_idx = filtered_with_idx[-1][0]

    if first_idx > 0:
        out_lines.append("   ... snipped")
    for _, line in filtered_with_idx:
        out_lines.append(line)
    if last_idx < len(section_lines) - 1:
        out_lines.append("   ... snipped")

    return True


def process_log(target: int):
    log_path = "logs/02_purified.txt"
    if not os.path.isfile(log_path):
        print(f"错误：日志文件 {log_path} 不存在")
        sys.exit(1)

    with open(log_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    min_id = max(0, target - RANGE)
    max_id = target + RANGE

    out_lines = []

    # 状态变量
    current_section = None      # 'brightness', 'reason', 'hash'
    pending_header = None
    section_lines = []
    in_cluster = False          # 是否已进入聚类部分（之后全部原样输出）

    # 三个部分的标题关键字
    section_headers = {
        "亮度序列 (帧名: 亮度值):": 'brightness',
        "每帧过滤原因:": 'reason',
        "保留帧的哈希值 (十六进制) 与拉普拉斯方差:": 'hash',
    }
    # 聚类部分标题（只要以该字符串开头即可）
    CLUSTER_HEADER = "聚类页面簇"

    def flush_section():
        """处理并输出当前累积的部分（仅在进入聚类前调用）"""
        nonlocal current_section, pending_header, section_lines
        if current_section is not None and section_lines:
            write_section(out_lines, pending_header, section_lines,
                          target, min_id, max_id)
        current_section = None
        pending_header = None
        section_lines = []

    for raw in raw_lines:
        line = remove_timestamp(raw).rstrip('\n')
        if not line:
            continue

        # 如果已经进入聚类部分，直接添加，不再做任何处理
        if in_cluster:
            out_lines.append(line)
            continue

        # 检查是否为聚类标题（一旦出现，立即切换到原样输出模式）
        if line.startswith(CLUSTER_HEADER):
            # 先处理之前累积的部分（亮度、原因、哈希）
            flush_section()
            # 输出聚类标题
            out_lines.append(line)
            in_cluster = True
            continue

        # 检查是否为三个部分的标题
        header_type = None
        for header, typ in section_headers.items():
            if line.startswith(header):
                header_type = typ
                break

        if header_type is not None:
            # 完成上一个部分
            flush_section()
            # 开始新部分
            pending_header = line
            current_section = header_type
            continue

        # 非标题行：如果在某个部分内，累积；否则直接输出（比如“=== 开始处理 ===”等）
        if current_section is not None:
            section_lines.append(line)
        else:
            out_lines.append(line)

    # 如果未进入聚类，处理最后可能累积的部分（正常情况下不会，因为聚类会触发切换）
    if not in_cluster:
        flush_section()

    # 写入输出文件
    out_dir = "logs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"02_purified_snip_{target}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"精简日志已保存到 {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python script.py <帧编号>")
        sys.exit(1)
    try:
        target_frame = int(sys.argv[1])
    except ValueError:
        print("帧编号必须是整数")
        sys.exit(1)
    process_log(target_frame)
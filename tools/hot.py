#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) < 2:
        print("用法: python script.py <匹配数字> [fps]")
        sys.exit(1)

    match_str = sys.argv[1]          # 例如 "15"
    fps = sys.argv[2] if len(sys.argv) > 2 else "1"  # 例如 "2"

    try:
        with open("data/ids.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 按等号分割，最多分割一次
                parts = line.split("=", 1)
                if len(parts) != 2:
                    continue
                title = parts[0].strip()
                id_val = parts[1].strip()
                # 检查标题中是否包含匹配字符串
                if match_str in title:
                    # 输出 ffmpeg 命令
                    print(f"ffmpeg -i video/av{id_val}.mp4 -vf fps={fps} cache/f{match_str}_%06d.png")
    except FileNotFoundError:
        print("错误: 未找到 data/ids 文件", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
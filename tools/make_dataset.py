import json
import os
import random
import shutil

SRC_DIR = "cache/crush-frames"
OUT_DIR = "dataset"

random.seed(42)

with open("cache/markframes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

items = []

for item in data:
    filename = os.path.basename(item["url"])
    image_path = os.path.join(SRC_DIR, filename)

    if not os.path.exists(image_path):
        continue

    label = "yes" if item["color"] == "red" else "no"

    items.append((image_path, label))

random.shuffle(items)

split = int(len(items) * 0.8)

train = items[:split]
val = items[split:]

for subset, dataset in [
    ("train", train),
    ("val", val),
]:
    for path, label in dataset:
        dst_dir = os.path.join(OUT_DIR, subset, label)
        os.makedirs(dst_dir, exist_ok=True)

        shutil.copy2(path, os.path.join(dst_dir, os.path.basename(path)))

print("Done")
import json
import os
import random
import shutil
from common import *
# SRC_DIR = "cache/frames"
# OUT_DIR = "dataset"

random.seed(42)

with open("cache/markframes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

yes = ClusterTable()
no = ClusterTable()

for item in data:
    name = os.path.basename(item["url"])

    if item['color'] == 'red':
        yes.add_frame(name)
    else:
        no.add_frame(name)

yes.save('cache/yes.json')
no.save('cache/no.json')

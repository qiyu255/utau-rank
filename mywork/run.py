from pathlib import Path
import glob
import pickle
import ch00_ready
import ch01_clean_daekness
import ch02_clean_miss
import ch03_dedup
import ch04_group
from common import *

CACHE_DIR = Path('cache/.pickle')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

pipeline = [
    ch00_ready,
    ch01_clean_daekness,
    ch02_clean_miss,
    ch03_dedup,
    # ch04_group,
]

def main():
    start = 3
    if start:
        with open(CACHE_DIR/(Path(pipeline[start-1].__file__)).stem, 'rb') as f:
            param, ctx = pickle.load(f)
    else:
        ctx = Context()
        param = [Path(i) for i in sorted(glob.glob('cache/frames/*.png'))]

    for p in pipeline[start:]:
        pname =  Path( p.__file__).stem

        hasattr(p, 'init') and p.init()
        param = p.run(param, ctx)
        hasattr(p, 'free') and p.free()

        with open(CACHE_DIR/pname, 'wb') as f:
            pickle.dump((param, ctx), f)


if __name__ == '__main__':
    main()

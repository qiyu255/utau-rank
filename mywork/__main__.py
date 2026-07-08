from common import *
from pathlib import Path
import numpy as np
import pandas as pd

logger.info('hello %s', __package__)
logger.info('hello %s', Path(__file__).stem)


def proc(view, n):
    print(view)
    view[0]+=n
    view[1]+=n
    view[2]+=n

m = np.ones((10,5),dtype=np.float64)

proc(m[0, 1:4], 2)

print(m)
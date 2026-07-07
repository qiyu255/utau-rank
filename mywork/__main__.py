from common import *
from pathlib import Path
import numpy as np
import pandas as pd

logger.info('hello %s', __package__)
logger.info('hello %s', Path(__file__).stem)


m = np.empty((3, 2), np.int8)

print(m)

# 2. 预设字段名
preset_columns = ['feature_1', 'feature_2', 'feature_3']
# 3. 预设 ID 列
preset_ids = ['i', '3'] # 生成 1001 到 1005 的 ID
# 4. 加载矩阵并设置预设字段名和 ID 列
df = pd.DataFrame(data=m, index=preset_ids, columns=preset_columns)
print(df)



import numpy as np
import pandas as pd

# 加载元数据
metadata = pd.read_csv("moe_server_data/metadata.csv")
print(metadata)

# 加载第一个张量
first_tensor = np.load("moe_server_data/" + metadata.iloc[0]["文件名"])
print(f"Shape: {first_tensor.shape}, Data: {first_tensor}")

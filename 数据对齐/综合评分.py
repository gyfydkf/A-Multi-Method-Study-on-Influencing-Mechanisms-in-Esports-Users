import pandas as pd
import numpy as np

def entropy_weight_method(df):
    X = df.values.astype(float)

    # 计算比例矩阵 P
    P = X / (X.sum(axis=0) + 1e-9)

    # 计算熵值
    E = -np.nansum(P * np.log(P + 1e-9), axis=0) / np.log(len(X))

    # 计算差异系数
    d = 1 - E

    # 计算权重
    w = d / d.sum()

    # 综合评分（直接用原始值）
    scores = X @ w

    return scores, w

# 读取数据
df = pd.read_csv('normalized_output.csv')

# 设置列名
time_col = 'date'
macro_cols = ['revenue','download','reviews','rating']

# 熵权法打分
scores, weights = entropy_weight_method(df[macro_cols])
df['综合评分'] = scores

# 保存为 CSV
df.to_csv('综合评分_每月.csv', index=False)

# 输出结果
print("✅ 每月综合评分如下：\n")
print(df[[time_col, '综合评分']])

print("\n📊 指标权重如下：")
for col, w in zip(macro_cols, weights):
    print(f"{col}: {w:.4f}")

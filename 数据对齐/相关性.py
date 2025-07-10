import pandas as pd

# 读取 CSV 文件（替换成你的文件路径）
df = pd.read_csv("normalized_output.csv")

# 查看列名，确认你要计算的两个变量
print(df.columns)

# 假设你要计算变量 'A' 和 'B' 之间的皮尔逊相关系数
correlation = df['综合评分1'].corr(df['综合评分2'])

print("相关系数 (Pearson):", correlation)

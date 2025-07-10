import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取原始CSV文件
df = pd.read_csv('222.csv')  # 替换为你的文件名

# 备份日期列（不参与归一化）
#date_column = df['date']
# 选取需要归一化的数值列
columns_to_normalize = ['第五','和平','王者']

# 创建归一化器
scaler = MinMaxScaler()

# 执行归一化
df_normalized = pd.DataFrame(scaler.fit_transform(df[columns_to_normalize]),
                             columns=columns_to_normalize)

# 恢复日期列
#df_normalized['date'] = date_column

# 如果需要将日期列放到第一列（可选）
#df_normalized = df_normalized[['date'] + columns_to_normalize]

# 保存为新的CSV文件
df_normalized.to_csv('normalized_output.csv', index=False)

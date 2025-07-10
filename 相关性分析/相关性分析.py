import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: 数据加载
folder_path = '/path/to/csv/folder'  # 替换为实际CSV文件夹路径
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 加载所有CSV文件
data_list = []
for file in csv_files:
    df = pd.read_csv(os.path.join(folder_path, file))
    data_list.append(df)

# 假设每个CSV文件中的列结构是一样的，将它们合并为一个大DataFrame
data = pd.concat(data_list, axis=1)

# Step 2: 多元线性回归与相关系数计算
X = data.iloc[:, :-1]  # 假设数据最后一列是目标变量
y = data.iloc[:, -1]   # 目标变量

# 拟合线性回归模型
model = LinearRegression()
model.fit(X, y)

# 获取回归系数（相关系数）
coefficients = model.coef_
coefficients_df = pd.DataFrame(coefficients, index=X.columns, columns=['Coefficient'])
print("多元线性回归的回归系数（相关系数）：")
print(coefficients_df)

# Step 3: 随机森林与特征重要性计算
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# 获取特征重要性
importances = rf_model.feature_importances_
importances_df = pd.DataFrame(importances, index=X.columns, columns=['Importance'])
print("\n随机森林的特征重要性：")
print(importances_df)

# Step 4: 主成分分析（PCA）降维
# 合并回归系数和特征重要性
combined_df = pd.DataFrame({
    'Coefficient': coefficients,
    'Importance': importances
}, index=X.columns)

# 标准化数据
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined_df)

# PCA降维
pca = PCA(n_components=1)  # 降维到一个主成分
pca_result = pca.fit_transform(combined_scaled)

# 将PCA评分添加到数据中
combined_df['PCA_Score'] = pca_result
print("\nPCA降维结果（综合评分）：")
print(combined_df)

# Step 5: 可视化
# 1. 绘制相关矩阵
corr_matrix = data.corr()

# 绘制相关矩阵热力图
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# 2. 绘制PCA评分的热力图
pca_score_df = pd.DataFrame(pca_result, index=X.columns, columns=['PCA_Score'])
plt.figure(figsize=(8, 6))
sns.heatmap(pca_score_df.T, annot=True, cmap='Blues', cbar=True)
plt.title('PCA Scores')
plt.show()


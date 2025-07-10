import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 假设df是你的数据集
# 如果你已经加载了数据，直接跳到以下步骤
df = pd.read_csv('hepin.csv')  # 如果需要加载数据

# 创建综合评分
df['user_love_score'] = (
    df['revenue'] * 0.25 + 
    df['download'] * 0.25 + 
    df['reviews'] * 0.25 + 
    df['rating'] * 0.25
)

# 可视化原始综合评分分布
plt.figure(figsize=(12, 8))
sns.histplot(df['user_love_score'], kde=True, bins=30)
plt.title('Distribution of User Love Score')
plt.xlabel('User Love Score')
plt.ylabel('Frequency')
plt.show()

# 选取相关特征进行PCA降维
features = df[['revenue', 'download', 'reviews', 'rating']]  # 选择用于PCA的特征

# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 应用PCA
pca = PCA(n_components=1)  # 选择1个主成分来替代综合评分
df['pca_user_love_score'] = pca.fit_transform(features_scaled)

# 可视化PCA降维后的得分分布
plt.figure(figsize=(12, 8))
sns.histplot(df['pca_user_love_score'], kde=True, bins=30, color='orange')
plt.title('Distribution of PCA User Love Score')
plt.xlabel('PCA User Love Score')
plt.ylabel('Frequency')
plt.show()

# 查看主成分的方差解释比例
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio of the principal component: {explained_variance[0]:.4f}")

# 可视化原始综合评分与PCA得分的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['user_love_score'], y=df['pca_user_love_score'], alpha=0.6)
plt.title('Original User Love Score vs PCA User Love Score')
plt.xlabel('Original User Love Score')
plt.ylabel('PCA User Love Score')
plt.show()

# 查看PCA主成分的系数
print(f"PCA component coefficients (loadings): {pca.components_}")

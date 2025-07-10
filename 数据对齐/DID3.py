import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white

# 假设df是你的数据集
df = pd.read_csv('wang.csv')  # 如果需要加载数据

from sklearn.preprocessing import MinMaxScaler

# 创建 MinMaxScaler 对象
scaler = MinMaxScaler()

# 对这四个字段进行归一化
df[['revenue', 'download', 'reviews', 'rating']] = scaler.fit_transform(df[['revenue', 'download', 'reviews', 'rating']])
'''
# 创建综合评分
df['user_love_score'] = (
    df['revenue'] * 0.25 + 
    df['download'] * 0.25 + 
    df['reviews'] * 0.25 + 
    df['rating'] * 0.25
)
'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
features = df[['reviews', 'rating','download','revenue']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=1)
df['user_love_score'] = pca.fit_transform(features_scaled)

'''
# 查看数据框的前几行
print(df)
'''

# 可视化各变量的分布
plt.figure(figsize=(12, 8))
sns.histplot(df['user_love_score'], kde=True, bins=30)
plt.title('Distribution of User Love Score')
plt.xlabel('User Love Score')
plt.ylabel('Frequency')
plt.show()

# 创建交互项：比赛月与比赛后
df['interaction'] = df['is_competition_month'] * df['post']

# 滞后项：比赛后的第1、2、3个月
df['lag1'] = df['user_love_score'].shift(1)  # 比赛后的第1个月
df['lag2'] = df['user_love_score'].shift(2)  # 比赛后的第2个月
df['lag3'] = df['user_love_score'].shift(3)  # 比赛后的第3个月

# 处理缺失值：删除包含NaN或Inf的行
df_cleaned = df.dropna()  # 删除含有缺失值的行

# 检查是否有无穷大值
df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)  # 替换无穷大值为NaN
df_cleaned = df_cleaned.dropna()  # 删除替换后的NaN值

# 查看滞后项
print(df_cleaned[['lag1', 'lag2', 'lag3']].head())

# 可视化滞后项的分布
plt.figure(figsize=(12, 8))
sns.histplot(df_cleaned['lag1'], kde=True, color='blue', label='Lag 1', bins=30)
sns.histplot(df_cleaned['lag2'], kde=True, color='green', label='Lag 2', bins=30)
sns.histplot(df_cleaned['lag3'], kde=True, color='red', label='Lag 3', bins=30)
plt.title('Distribution of Lagged User Love Scores')
plt.xlabel('Lagged User Love Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# --------- 👇 平行趋势图：分组趋势可视化 👇 ---------

# 确保你有“时间”这一列，比如 ‘month’ 或者你自定义的时间变量
# 如果没有month列，可以用df.index或自己造一个
# 例如：df['month'] = range(len(df))

# 首先确保你的数据中有代表时间的变量
if 'month' not in df.columns:
    df['month'] = range(len(df))  # 如果没有month列，临时造一个

# 分组趋势图：比较处理组 vs 对照组在处理前后的平均变化
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='month', y='user_love_score', hue='is_competition_month', style='post', markers=True)
plt.title('User Love Score Trends: Treatment vs Control Groups')
plt.xlabel('Time (Month Index)')
plt.ylabel('User Love Score')
plt.legend(title='Group')
plt.grid(True)
plt.show()

# --------- 👇 更标准的 DID 模型 👇 ---------

# 仅使用DID变量：比赛组、比赛后、交互项
X_did = sm.add_constant(df_cleaned[['is_competition_month', 'post', 'interaction']])
y = df_cleaned['user_love_score']

model_did = sm.OLS(y, X_did).fit()
print(model_did.summary())

# 可视化回归系数（更简洁）
plt.figure(figsize=(8, 5))
sns.barplot(x=model_did.params.index, y=model_did.params.values)
plt.title('DID Model Coefficients')
plt.ylabel('Coefficient')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 构建自变量，包含比赛月、比赛后、交互项和滞后项
X = df_cleaned[['is_competition_month', 'post', 'interaction', 'lag1', 'lag2', 'lag3']]
X = sm.add_constant(X)  # 添加常数项
y = df_cleaned['user_love_score']  # 因变量：用户喜爱度的综合评分

# 回归分析
model = sm.OLS(y, X).fit()

# 输出回归结果
print(model.summary())

# 可视化回归系数
plt.figure(figsize=(10, 6))
sns.barplot(x=model.params.index, y=model.params.values)
plt.title('Regression Coefficients')
plt.xlabel('Variables')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.show()

# 多重共线性检验：VIF（方差膨胀因子）
# 去掉常数项
X_vif = X.drop(columns=['const'])

# 计算VIF
vif_data = pd.DataFrame()
vif_data['Variable'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# 输出VIF数据
print(vif_data)

# 可视化VIF
plt.figure(figsize=(10, 6))
sns.barplot(x=vif_data['Variable'], y=vif_data['VIF'])
plt.title('Variance Inflation Factor (VIF) for Variables')
plt.xlabel('Variable')
plt.ylabel('VIF')
plt.xticks(rotation=45)
plt.show()

# 异方差性检验（White Test）
# 计算White检验的p值
white_test = het_white(model.resid, model.model.exog)
print(f"White Test p-value: {white_test[1]}")

# 可视化残差与拟合值
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model.fittedvalues, y=model.resid)
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# 打印滞后项系数
print(f"lag1 coefficient: {model.params['lag1']}")
print(f"lag2 coefficient: {model.params['lag2']}")
print(f"lag3 coefficient: {model.params['lag3']}")

# 可视化滞后项系数
plt.figure(figsize=(10, 6))
sns.barplot(x=['lag1', 'lag2', 'lag3'], y=[model.params['lag1'], model.params['lag2'], model.params['lag3']])
plt.title('Lagged Effects on User Love Score')
plt.xlabel('Lag Period')
plt.ylabel('Coefficient Value')
plt.show()

# 输出模型拟合优度
print(f"R-squared: {model.rsquared}")

# 可视化真实值与预测值的比较
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=model.fittedvalues)
plt.title('Observed vs Predicted User Love Score')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.show()

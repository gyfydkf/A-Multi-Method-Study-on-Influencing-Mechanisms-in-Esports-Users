import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white
from sklearn.preprocessing import MinMaxScaler

# 1. 加载数据
df = pd.read_csv('wangzhe.csv')

# 2. 归一化字段
scaler = MinMaxScaler()
df[['revenue', 'download', 'reviews', 'rating']] = scaler.fit_transform(df[['revenue', 'download', 'reviews', 'rating']])

# 3. 构建用户喜爱度得分
df['user_love_score'] = (
    df['revenue'] * 0.25 +
    df['download'] * 0.25 +
    df['reviews'] * 0.25 +
    df['rating'] * 0.25
)

# 4. 构造处理变量 treatment（假设有 game_id 标识游戏）
df['treatment'] = df.groupby('game_id')['is_competition_month'].transform('max')

# 5. 构造 DID 交互项
df['did'] = df['treatment'] * df['post']

# 6. 滞后项（用于控制动态趋势）
df['lag1'] = df['user_love_score'].shift(1)
df['lag2'] = df['user_love_score'].shift(2)
df['lag3'] = df['user_love_score'].shift(3)

# 7. 删除缺失值和无穷大
df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna()

# 8. 回归模型：DID + 滞后控制
X = df_cleaned[['treatment', 'post', 'did', 'lag1', 'lag2', 'lag3']]
X = sm.add_constant(X)
y = df_cleaned['user_love_score']

model = sm.OLS(y, X).fit(cov_type='HC3')  # 使用稳健标准误

# 9. 打印结果
print(model.summary())
print(f"\nDID交互项系数（did） = {model.params['did']:.4f}")
print("解释：该系数衡量比赛对用户喜爱度的因果影响")

# 10. 可视化回归系数
plt.figure(figsize=(10, 6))
sns.barplot(x=model.params.index, y=model.params.values)
plt.title('回归系数（含DID效应）')
plt.xticks(rotation=45)
plt.show()

# 11. VIF检验
X_vif = X.copy()
if 'const' in X_vif.columns:
    X_vif = X_vif.drop(columns='const')

#X_vif = X.drop(columns='const')
vif_data = pd.DataFrame()
vif_data['Variable'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)

# 12. 异方差检验
white_test = het_white(model.resid, model.model.exog)
print(f"White Test p-value: {white_test[1]:.4f}")

# 13. 残差图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model.fittedvalues, y=model.resid)
plt.title('残差 vs 拟合值')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# 14. 真实值 vs 预测值
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=model.fittedvalues)
plt.title('真实值 vs 预测值')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.show()

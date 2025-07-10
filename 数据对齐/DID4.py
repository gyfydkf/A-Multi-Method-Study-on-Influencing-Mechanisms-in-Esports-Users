import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv('hepin.csv', parse_dates=['date'])

# 1. 主成分分析构建“用户喜爱度”指标
features = df[['reviews', 'rating','download','revenue']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=1)
df['love_score'] = pca.fit_transform(features_scaled)

# 2. 构造 DID 交互项（比赛月 × 后期）
df['did'] = df['is_competition_month'] * df['post']

# 3. 回归分析：用户喜爱度 ~ 比赛月 + 后期 + 交互项
X = df[['is_competition_month', 'post', 'did']]
X = sm.add_constant(X)
y = df['love_score']

model = sm.OLS(y, X).fit()
print(model.summary())

# 4. 可视化趋势图
df['year_month'] = df['date'].dt.to_period('M').astype(str)

plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='year_month', y='love_score', hue='is_competition_month', palette='Set1')
plt.xticks(rotation=45)
plt.title("用户喜爱度：比赛月 vs 非比赛月")
plt.xlabel("日期")
plt.ylabel("用户喜爱度（PCA）")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# 读取数据
df = pd.read_csv('game.csv')

# 归一化
scaler = MinMaxScaler()
df[['revenue', 'download', 'reviews', 'rating']] = scaler.fit_transform(df[['revenue', 'download', 'reviews', 'rating']])

# PCA 综合评分
features = df[['reviews', 'rating','download','revenue']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=1)
df['user_love_score'] = pca.fit_transform(features_scaled)

# 📊 用户喜爱度分布图（冷色调）
fig = px.histogram(df, x='user_love_score', nbins=30, marginal="rug", 
                   title='Distribution of User Love Score',
                   labels={'user_love_score': 'User Love Score'}, 
                   opacity=0.75,
                   color_discrete_sequence=px.colors.sequential.Mint)#Emrld黄绿，Blugrn青绿，Peach橙色
fig.update_layout(bargap=0.15, template='plotly_white', 
                  title_font_size=20, font=dict(size=14),
                  plot_bgcolor='rgba(245, 250, 255, 1)')
fig.show()

# 创建交互项 & 滞后变量
df['interaction'] = df['is_competition_month'] * df['post']
df['lag1'] = df['user_love_score'].shift(1)
df['lag2'] = df['user_love_score'].shift(2)
df['lag3'] = df['user_love_score'].shift(3)

# 清洗数据
df_cleaned = df.dropna()
df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan).dropna()

# 📊 滞后项分布图（夕阳色调）
fig = px.histogram(df_cleaned.melt(value_vars=['lag1', 'lag2', 'lag3'], 
                                   var_name='Lag', value_name='Score'),
                   x='Score', color='Lag', marginal="violin", barmode='overlay',
                   nbins=30, title='Distribution of Lagged User Love Scores',
                   color_discrete_sequence=px.colors.sequential.Sunset)
fig.update_traces(opacity=0.6)
fig.update_layout(template='ggplot2', font=dict(size=13))
fig.show()

# 📈 平行趋势图（淡绿柔蓝）
if 'month' not in df.columns:
    df['month'] = range(len(df))

fig = px.line(df, x='month', y='user_love_score', 
              color='is_competition_month', 
              line_dash='post', 
              markers=True,
              color_discrete_sequence=px.colors.sequential.Magenta,
              title='User Love Score Trends: Treatment vs Control Groups',
              labels={
                  'month': 'Time (Month Index)', 
                  'user_love_score': 'User Love Score',
                  'is_competition_month': 'Competition Group',
                  'post': 'After Competition'
              })
fig.update_layout(template='simple_white', title_font_size=18,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
fig.show()

# 🎯 DID 模型
X_did = sm.add_constant(df_cleaned[['is_competition_month', 'post', 'interaction']])
y = df_cleaned['user_love_score']
model_did = sm.OLS(y, X_did).fit()
print(model_did.summary())

# 📊 DID系数图（薄荷绿调）
fig = px.bar(x=model_did.params.index, y=model_did.params.values,
             title='DID Model Coefficients',
             labels={'x': 'Variables', 'y': 'Coefficient'},
             color=model_did.params.index,
             color_discrete_sequence=px.colors.sequential.Mint)
fig.update_layout(showlegend=False, template='plotly_white', xaxis_tickangle=-15)
fig.show()

# 🎯 完整模型（带滞后项）
X = df_cleaned[['is_competition_month', 'post', 'interaction', 'lag1', 'lag2', 'lag3']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# 📊 回归系数图（多彩花园）
fig = px.bar(x=model.params.index, y=model.params.values,
             title='Regression Coefficients',
             labels={'x': 'Variables', 'y': 'Coefficient'},
             color=model.params.index,
             color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(template='seaborn', xaxis_tickangle=-45, showlegend=False)
fig.show()

# 🔍 VIF 检验
X_vif = X.drop(columns=['const'])
vif_data = pd.DataFrame()
vif_data['Variable'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)

# 📊 VIF 可视化图（科幻蓝紫）
fig = px.bar(vif_data, x='Variable', y='VIF',
             title='Variance Inflation Factor (VIF) for Variables',
             color='VIF',
             color_continuous_scale=px.colors.sequential.Purples)
fig.update_layout(template='plotly_dark', xaxis_tickangle=-20)
fig.show()

# 🔍 异方差性检验
white_test = het_white(model.resid, model.model.exog)
print(f"White Test p-value: {white_test[1]}")

# 📊 残差 vs 拟合值图（温柔米白背景）
fig = px.scatter(x=model.fittedvalues, y=model.resid,
                 labels={'x': 'Fitted Values', 'y': 'Residuals'},
                 title='Residuals vs Fitted Values',
                 color=model.fittedvalues,
                 color_continuous_scale=px.colors.sequential.Burg)
fig.update_traces(marker=dict(size=7, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(template='simple_white')
fig.show()

# 🔁 滞后项系数
print(f"lag1 coefficient: {model.params['lag1']}")
print(f"lag2 coefficient: {model.params['lag2']}")
print(f"lag3 coefficient: {model.params['lag3']}")

# 📊 滞后项系数图（黄绿对比）
lags = ['lag1', 'lag2', 'lag3']
coeffs = [model.params['lag1'], model.params['lag2'], model.params['lag3']]
fig = px.bar(x=lags, y=coeffs,
             title='Lagged Effects on User Love Score',
             labels={'x': 'Lag Period', 'y': 'Coefficient Value'},
             color=lags,
             color_discrete_sequence=px.colors.sequential.YlGn)
fig.update_layout(template='ggplot2', showlegend=False)
fig.show()

# 📈 真实值 vs 预测值图（海洋蓝 + 参考线）
print(f"R-squared: {model.rsquared}")
fig = px.scatter(x=y, y=model.fittedvalues,
                 labels={'x': 'Observed', 'y': 'Predicted'},
                 title='Observed vs Predicted User Love Score',
                 color=y,
                 color_continuous_scale=px.colors.sequential.Blues)
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.update_layout(template='plotly_white')
fig.show()

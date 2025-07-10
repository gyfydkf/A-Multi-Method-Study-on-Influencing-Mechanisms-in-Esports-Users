import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white

# 读取数据
df = pd.read_csv('wang.csv')

# 使用已有的用户喜爱度列
# 假设 CSV 中列名为 'user_love'
df['user_love_score'] = df['user_love']

# 📊 用户喜爱度分布图（冷色调）
fig = px.histogram(
    df,
    x='user_love_score',
    nbins=30,
    marginal='rug',
    title='Distribution of User Love Score',
    labels={'user_love_score': 'User Love Score'},
    opacity=0.75,
    color_discrete_sequence=px.colors.sequential.Darkmint
)
fig.update_layout(
    bargap=0.15,
    template='plotly_white',
    title_font_size=20,
    font=dict(size=14),
    plot_bgcolor='rgba(245, 250, 255, 1)'
)
fig.show()

# 创建交互项 & 滞后变量
df['interaction'] = df['is_competition_month'] * df['post']
for lag in range(1, 4):
    df[f'lag{lag}'] = df['user_love_score'].shift(lag)

# 清洗数据
df_clean = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()

# 📊 滞后项分布图（夕阳色调）
lag_cols = ['lag1', 'lag2', 'lag3']
df_melt = df_clean.melt(value_vars=lag_cols, var_name='Lag', value_name='Score')
fig = px.histogram(
    df_melt,
    x='Score',
    color='Lag',
    marginal='violin',
    barmode='overlay',
    nbins=30,
    title='Distribution of Lagged User Love Scores',
    color_discrete_sequence=px.colors.sequential.Blugrn
)
fig.update_traces(opacity=0.6)
fig.update_layout(template='ggplot2', font=dict(size=13))
fig.show()

# 📈 平行趋势图（淡绿柔蓝）
if 'month' not in df.columns:
    df['month'] = range(len(df))
fig = px.line(
    df,
    x='month',
    y='user_love_score',
    color='is_competition_month',
    line_dash='post',
    markers=True,
    color_discrete_sequence=px.colors.sequential.Emrld,
    title='User Love Score Trends: Treatment vs Control Groups',
    labels={
        'month': 'Time (Month Index)',
        'user_love_score': 'User Love Score',
        'is_competition_month': 'Competition Group',
        'post': 'After Competition'
    }
)
fig.update_layout(
    template='simple_white',
    title_font_size=18,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    )
)
fig.show()

# 🎯 DID 模型
X_did = sm.add_constant(
    df_clean[['is_competition_month', 'post', 'interaction']]
)
y_did = df_clean['user_love_score'].loc[X_did.index]
model_did = sm.OLS(y_did, X_did).fit()
print(model_did.summary())

# 📊 DID系数图（薄荷绿调）
fig = px.bar(
    x=model_did.params.index,
    y=model_did.params.values,
    title='DID Model Coefficients',
    labels={'x': 'Variables', 'y': 'Coefficient'},
    color=model_did.params.index,
    color_discrete_sequence=px.colors.sequential.Teal
)
fig.update_layout(showlegend=False, template='plotly_white', xaxis_tickangle=-15)
fig.show()

# 🎯 完整模型（带滞后项）
X_full = df_clean[['is_competition_month', 'post', 'interaction'] + lag_cols]
X_full = sm.add_constant(X_full)
y_full = df_clean['user_love_score']
model_full = sm.OLS(y_full, X_full).fit()
print(model_full.summary())

# 📊 回归系数图（多彩花园）
fig = px.bar(
    x=model_full.params.index,
    y=model_full.params.values,
    title='Regression Coefficients',
    labels={'x': 'Variables', 'y': 'Coefficient'},
    color=model_full.params.index,
    color_discrete_sequence=px.colors.diverging.Tealrose
)
fig.update_layout(template='seaborn', xaxis_tickangle=-45, showlegend=False)
fig.show()

# 🔍 VIF 检验
X_vif = X_full.drop(columns=['const'])
vif_df = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
})
print(vif_df)

# 📊 VIF 可视化图（科幻蓝紫）
fig = px.bar(
    vif_df,
    x='Variable',
    y='VIF',
    title='Variance Inflation Factor (VIF) for Variables',
    color='VIF',
    color_continuous_scale=px.colors.sequential.Teal
)
fig.update_layout(template='plotly_white', xaxis_tickangle=-20)
fig.show()

# 🔍 异方差性检验
white_test = het_white(model_full.resid, model_full.model.exog)
print(f"White Test p-value: {white_test[1]}")

# 📊 残差 vs 拟合值图（温柔米白背景）
fig = px.scatter(
    x=model_full.fittedvalues,
    y=model_full.resid,
    labels={'x': 'Fitted Values', 'y': 'Residuals'},
    title='Residuals vs Fitted Values',
    color=model_full.fittedvalues,
    color_continuous_scale=px.colors.sequential.Teal
)
fig.update_traces(marker=dict(size=7, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(template='plotly_white')
fig.show()

# 🔁 滞后项系数
for lag in lag_cols:
    print(f"{lag} coefficient: {model_full.params[lag]}")

# 📊 滞后项系数图（黄绿对比）
coeffs = [model_full.params[lag] for lag in lag_cols]
fig = px.bar(
    x=lag_cols,
    y=coeffs,
    title='Lagged Effects on User Love Score',
    labels={'x': 'Lag Period', 'y': 'Coefficient Value'},
    color=lag_cols,
    color_discrete_sequence=px.colors.sequential.Greens
)
fig.update_layout(template='ggplot2', showlegend=False)
fig.show()

# 📈 真实值 vs 预测值图（海洋蓝 + 参考线）
print(f"R-squared: {model_full.rsquared}")
fig = px.scatter(
    x=y_full,
    y=model_full.fittedvalues,
    labels={'x': 'Observed', 'y': 'Predicted'},
    title='Observed vs Predicted User Love Score',
    color=y_full,
    color_continuous_scale=px.colors.sequential.Burg
)
fig.add_trace(go.Scatter(
    x=[y_full.min(), y_full.max()],
    y=[y_full.min(), y_full.max()],
    mode='lines',
    line=dict(dash='dash', color='gray'),
    showlegend=False
))
fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.update_layout(template='plotly_white')
fig.show()

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1：读取数据
df = pd.read_csv('d.csv')  # 把文件名替换为你的实际路径

# Step 2：处理日期
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Step 3：残差建模 —— 构建预期下载模型，剔除宏观行业因素
model_base = smf.ols("downlode ~ 预估下载 + 市场规模 + 游戏数量 + 用户规模 + year + month", data=df).fit()
df["residual"] = df["downlode"] - model_base.fittedvalues

# 你可以查看模型的摘要：
print(model_base.summary())

# Step 4：DID分析 —— 比赛是否带来表现提升
# treatment（是否参加比赛） * post（比赛后） 的交互项
model_did = smf.ols("residual ~ is_competition_month * post + year + month", data=df).fit()
print(model_did.summary())

# DID 的核心变量是 is_competition_month:post，解释为比赛带来的净提升效果

# Step 5：滞后分析 —— 比赛后的1~3个月是否有持续效果
model_lag = smf.ols("residual ~ is_competition_month + lag1 + lag2 + lag3 + year + month", data=df).fit()
print(model_lag.summary())

# Step 6：可视化比赛前后表现变化趋势（事件研究图）
# 构造事件时间：-3 到 +3
df['event_time'] = (
    df['lag3'] * (-3) + df['lag2'] * (-2) + df['lag1'] * (-1) +
    df['is_competition_month'] * 0 +
    df['post'] * 1  # 这里是简化模型，如你有更多 post1, post2 可进一步细化
)

# 平均每个event_time的表现
event_df = df[df['is_competition_month'] == 1]  # 只选参加比赛的游戏
plot_data = event_df.groupby('event_time')['residual'].mean().reset_index()

# 画图
plt.figure(figsize=(8, 5))
sns.lineplot(data=plot_data, x='event_time', y='residual', marker='o')
plt.axvline(0, color='red', linestyle='--', label='比赛发生月')
plt.title("比赛前后游戏表现（残差）变化趋势")
plt.xlabel("比赛相对月份")
plt.ylabel("残差（实际下载 - 预期下载）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

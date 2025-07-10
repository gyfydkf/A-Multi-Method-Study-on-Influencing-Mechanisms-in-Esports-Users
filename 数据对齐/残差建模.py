import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. 读取CSV文件
df = pd.read_csv('normalized_output.csv')  # 替换为你的CSV文件路径

# 2. 检查列名
print(df.columns)

# 3. 指定变量
y = df['Y_original']                 # 原始目标变量
X = sm.add_constant(df['zp'])  # 加入截距项，'综合评分2' 是宏观因素

# 4. 拟合OLS
model = sm.OLS(y, X).fit()
print(model.summary())

# 5. 计算预测值（宏观影响部分）和“剔除宏观影响后的Y”
df['Y_macro_pred'] = model.predict(X)
df['Y_adjusted']   = df['Y_original'] - df['Y_macro_pred']

# 6. 可视化对比
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Y_original'],    label='Original Y',               linestyle='--')
plt.plot(df.index, df['Y_adjusted'],   label='Adjusted Y ', linewidth=2)
plt.legend()
#plt.title('剔除宏观影响后的 Y 对比原始 Y')
#plt.xlabel('时间索引或编号')
#plt.ylabel('数值')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. 保存结果：包括原始Y、预测的宏观部分、以及剔除后的Y
df.to_csv('residual_and_adjusted_output.csv', index=False)

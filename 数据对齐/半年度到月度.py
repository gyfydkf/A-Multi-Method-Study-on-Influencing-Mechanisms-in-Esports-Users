import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator  # ✅ 用 PCHIP

# 读取半年度数据
file_path = '中国游戏用户规模（全部）.csv'  # 改成你的文件路径
df = pd.read_csv(file_path, parse_dates=['date'])

# 确保date排序
df = df.sort_values('date')

# 构建数字化date轴（比如：2020-06 → 2020.5）
df['date数值'] = df['date'].dt.year + (df['date'].dt.month - 1) / 12

# 插值目标：从第一个月到最后一个月，每个月一个点
monthly_time_values = np.arange(df['date数值'].min(), df['date数值'].max() + 1/12, 1/12)

# 用 PCHIP 插值拟合 ✅
spline = PchipInterpolator(df['date数值'], df['Userscale'])
monthly_values = spline(monthly_time_values)

# 生成新的 DataFrame
monthly_dates = pd.date_range(start=df['date'].min(), periods=len(monthly_time_values), freq='MS')

monthly_df = pd.DataFrame({
    'date': monthly_dates,
    'Userscale': monthly_values
})

# 可视化对比一下
plt.figure(figsize=(12, 6))
plt.plot(monthly_df['date'], monthly_df['Userscale'], label='月度估算值 (PCHIP)', color='blue')
plt.scatter(df['date'], df['Userscale'], color='red', label='原始半年度数据')
plt.legend()
plt.title("半年度数据拆分到月度 (PCHIP 插值)")
plt.xlabel("date")
plt.ylabel("Userscale")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 保存结果
monthly_df.to_csv("半年度转月度结果.csv", index=False, encoding='gbk')

print("拆分完成！结果保存在：半年度转月度结果.csv")

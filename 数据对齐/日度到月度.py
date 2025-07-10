import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = 'd.csv'  # 替换成你的路径
df = pd.read_csv(file_path, parse_dates=['date'])
import pandas as pd

# 自动读取文件，尝试多种编码
# 打印数据的前几行，确认是否有问题
print(df.head())
# 确保数据按date排序
df = df.sort_values('date')

df['sums'] = pd.to_numeric(df['sums'], errors='coerce')

# 方法二：按月求总和（当月总用户量）
monthly_sum = df.resample('M', on='date').sum().reset_index()
monthly_sum.rename(columns={'sums': 'monthsums'}, inplace=True)
# 保存结果
monthly_sum.to_csv('monthsum.csv', index=False, encoding='gbk')

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(monthly_sum['date'], monthly_sum['monthsums'], marker='s', label='monthsums', alpha=0.7)
plt.legend()
plt.title("日度数据汇总为月度数据")
plt.xlabel("月份")
plt.ylabel("sum")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

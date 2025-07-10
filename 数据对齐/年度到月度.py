import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from statsmodels.tsa.seasonal import STL

# 读取数据（自动识别格式）
def read_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, delimiter="\t")
    else:
        raise ValueError("不支持的文件格式")
    return df

# 拆分到月份
def disaggregate_to_month(df):
    # 准备时间轴
    years = df['年份'].values
    values = df['用户规模'].values

    # 创建月度时间点（比如：2015.00, 2015.08, ..., 2023.92）
    months = np.arange(years.min(), years.max() + 1, 1/12)

    # 插值模型：立方样条
    spline = CubicSpline(years, values)
    monthly_values = spline(months)

    # 转成 DataFrame
    monthly_df = pd.DataFrame({
        '时间': pd.date_range(start=f'{years.min()}-01', periods=len(months), freq='MS'),
        '用户规模': monthly_values
    })

    return monthly_df

# 可选：做季节性分解，检查季节性趋势
def seasonal_decompose(monthly_df):
    stl = STL(monthly_df['用户规模'], period=12)
    result = stl.fit()
    result.plot()
    plt.suptitle("季节性分解结果")
    plt.show()

    return result

# 主程序
if __name__ == "__main__":
    file_path = '你的文件路径.csv'  # <- 改成你的文件路径
    df = read_data(file_path)
    monthly_df = disaggregate_to_month(df)

    print(monthly_df.head(12))  # 看看头12个月数据

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_df['时间'], monthly_df['用户规模'], label='月度用户规模')
    plt.scatter(df['年份'].astype(str) + "-01", df['用户规模'], color='red', label='原始年度数据')
    plt.legend()
    plt.title("年度数据拆分到月度")
    plt.xlabel("时间")
    plt.ylabel("用户规模")
    plt.show()

    # 可选：季节性分解分析
    seasonal_decompose(monthly_df)

    # 保存结果
    monthly_df.to_csv("月度用户规模预测.csv", index=False, encoding='utf-8-sig')

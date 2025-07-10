import pandas as pd

def count_ip_regions():
    try:
        # 读取文件
        df = pd.read_csv('merged_output.csv')

        # 检查数据集中是否有 'ip' 列
        if 'ip' not in df.columns:
            print("错误：数据集中不存在 'ip' 列。")
            return

        # 统计各个地区出现的数量
        region_counts = df['ip'].value_counts().reset_index(name='数量')

        # 保存为 csv 文件
        region_counts.to_csv('ip.csv', index=False)
        print("成功统计并保存到 ip.csv")

    except FileNotFoundError:
        print("错误：未找到文件 weibo.csv。")
    except Exception as e:
        print(f"发生未知错误：{e}")

if __name__ == "__main__":
    count_ip_regions()
    
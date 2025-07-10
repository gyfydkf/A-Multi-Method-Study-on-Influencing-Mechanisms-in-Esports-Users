import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取两个 CSV 文件
macro_df = pd.read_csv('hongguan.csv')  # 包含实际销售收入、预估下载、市场规模、游戏数量、用户规模
game_df = pd.read_csv('game_data.csv')    # 包含 revenue, download, reviews, rating

# 合并两个数据集（假设它们是一一对应的，按行合并）
combined_df = pd.concat([macro_df, game_df], axis=1)

# 计算相关系数矩阵
corr_matrix = combined_df.corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
#plt.title('相关系数热力图')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
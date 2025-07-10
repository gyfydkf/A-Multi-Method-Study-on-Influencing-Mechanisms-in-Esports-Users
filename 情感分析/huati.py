import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# 读取数据
file_path = 'merged_output.csv'  # 本地路径，根据实际情况修改
df = pd.read_csv(file_path)

# 查看数据列，确保列名正确
print(df.columns)

# 提取 topic 列，并去掉缺失值
topics = df['topics'].dropna()

# 统计每个话题的出现次数
topic_counts = Counter(topics)

# 转成 DataFrame
topic_df = pd.DataFrame(topic_counts.items(), columns=['Topic', 'Count'])
topic_df = topic_df.sort_values(by='Count', ascending=False)

# 保存为 CSV 文件
topic_df.to_csv('话题热度统计.csv', index=False, encoding='utf-8-sig')
print("已保存为 '话题热度统计.csv' 文件！")

# 打印 Top 10 热门话题
print("Top 10 热门话题：")
print(topic_df.head(10))

# 可视化：柱状图
plt.figure(figsize=(12, 6))
plt.bar(topic_df['Topic'][:10], topic_df['Count'][:10], color='skyblue')
plt.title('Top 10 热门话题')
plt.xlabel('话题')
plt.ylabel('出现次数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Top10热门话题.png', dpi=300)
plt.show()

# 可视化：词云
font_path = 'C:/Windows/Fonts/msyh.ttc'  # 改成你电脑里的字体文件路径
wordcloud = WordCloud(font_path=font_path,
                      width=800, height=400,
                      background_color='white').generate_from_frequencies(topic_counts)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('话题词云.png', dpi=300)
plt.show()

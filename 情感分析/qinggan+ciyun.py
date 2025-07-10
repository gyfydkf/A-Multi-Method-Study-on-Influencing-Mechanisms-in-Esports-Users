import pandas as pd
from snownlp import SnowNLP
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba

# 1. 读取数据
file_path = 'merged_output.csv'
df = pd.read_csv(file_path)

# 检查列名，假设文本列是 "text"
print(df.columns)

# 2. 情感分析函数
def analyze_sentiment(text):
    try:
        s = SnowNLP(text)
        return s.sentiments  # 返回 0~1, 越接近1越正面
    except:
        return None

# 3. 应用情感分析
print("开始情感分析...")
df['sentiment'] = df['text'].astype(str).apply(analyze_sentiment)

# 4. 统计情感结果
def sentiment_label(score):
    if score >= 0.6:
        return '正面'
    elif score <= 0.4:
        return '负面'
    else:
        return '中性'

df['sentiment_label'] = df['sentiment'].apply(sentiment_label)

# 保存仅包含 text 和情感结果的文件（确保只有两列）
output_df = df[['text', 'sentiment_label']].copy()
output_df.to_csv('weibo_sentiment_results.csv', index=False, encoding='utf-8-sig')
print("情感分析结果已保存为 weibo_sentiment_results.csv (仅包含 text 和情感标签)")

# 5. 绘制情感分布图
sentiment_counts = df['sentiment_label'].value_counts()

# 打印情感分析数量结果
print("情感分析数量统计：")
print(sentiment_counts)

# 可视化
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('微博情感分布')
plt.xlabel('情感')
plt.ylabel('数量')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.close()
print("情感分布图已保存为 sentiment_distribution.png")

'''
# 6. 生成词云
import jieba
from PIL import Image
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

# 分词处理
text = ' '.join(df['text'].astype(str))
wordlist = jieba.cut(text)
word_space_split = ' '.join(wordlist)

# 加载云状蒙版图
mask_image = np.array(Image.open('OIP.jpg'))    

# 创建词云对象
wordcloud = WordCloud(
    font_path='C:/Windows/Fonts/msyh.ttc',  # 中文字体路径，防止乱码
    background_color='white',
    mask=mask_image,
    contour_width=2,
    contour_color='orange',  # 云朵的轮廓色，橙色更亮眼
    colormap='autumn',  # 暖色系渐变色，橙-黄-红
    max_font_size=150,  # 增加最大字体大小
    random_state=42
).generate(word_space_split)

# 绘制词云
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('cloud_wordcloud_bright.png', dpi=300)
plt.close()
print("高亮云朵词云已保存为 cloud_wordcloud_bright.png")



print("生成词云...")
text = ' '.join(df['text'].astype(str))
wordlist = jieba.cut(text)
word_space_split = ' '.join(wordlist)

wordcloud = WordCloud(font_path='simhei.ttf', background_color='white', width=800, height=600).generate(word_space_split)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud.png')
plt.close()
print("词云图已保存为 wordcloud.png")

'''
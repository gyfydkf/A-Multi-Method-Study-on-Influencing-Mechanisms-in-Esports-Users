# 6. 生成词云
import jieba
import pandas as pd
from PIL import Image
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

# 分词处理
file_path = 'merged_output.csv'
df = pd.read_csv(file_path)
text = ' '.join(df['text'].astype(str))
wordlist = jieba.cut(text)
word_space_split = ' '.join(wordlist)

# 加载云状蒙版图
mask_image = np.array(Image.open('1.jpg'))    

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


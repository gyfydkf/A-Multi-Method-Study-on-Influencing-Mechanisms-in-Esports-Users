import pandas as pd
import jieba
from collections import Counter

# 读取数据
file_path = 'merged_output.csv'
data = pd.read_csv(file_path)

# 假设微博内容在 'content' 这一列，请根据实际列名修改
text_data = data['text'].astype(str)

# 合并所有文本
all_text = ' '.join(text_data)

# 使用 jieba 分词
words = jieba.lcut(all_text)

# 过滤掉长度为 1 的词语，避免无意义词
filtered_words = [word for word in words if len(word) > 1]

# 统计词频
word_counts = Counter(filtered_words)

# 转为 DataFrame
word_count_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])

# 按照频率排序
word_count_df = word_count_df.sort_values(by='Frequency', ascending=False)

# 保存结果到 CSV 文件
output_path = 'word_count.csv'
word_count_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f'词频统计完成，结果已保存到 {output_path}')

import pandas as pd
import jieba
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models
import re
import webbrowser
import os

# 配置文件路径
file_path = 'merged_output.csv'
stopwords_path = 'cn_stopwords.txt'  # 请提前下载好，放在同目录

# 1. 读取数据，检测编码
def read_csv_safely(file_path):
    for enc in ['utf-8', 'utf-8-sig', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"✅ 使用编码 {enc} 成功读取数据，共 {len(df)} 行")
            print(df.head())
            return df
        except Exception as e:
            print(f"⚠️ 尝试使用编码 {enc} 读取失败: {e}")
    print("❌ 所有编码方式读取失败，请检查文件格式！")
    exit()

df = read_csv_safely(file_path)

# 检查 'text' 列是否存在
if 'text' not in df.columns:
    print("❌ 原始数据中不存在 'text' 列，请检查列名是否正确！")
    exit()

# 2. 文本预处理：去除缺失值，清洗文本
texts = df['text'].dropna().astype(str).tolist()
print(f"✅ 删除缺失值后文本数量：{len(texts)}")

def clean_text(text):
    # 保留中文、英文和数字，去除其他符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text

texts = [clean_text(text) for text in texts]
print(f"✅ 文本清洗后示例：{texts[:5]}")

# 3. 分词处理
tokenized_texts = [list(jieba.cut(text)) for text in texts]
print(f"✅ 分词后示例：{tokenized_texts[:5]}")

# 4. 停用词处理（本地文件）
def load_stopwords(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ 停用词文件未找到：{filepath}，请确认文件路径！")
        return set()
    try:
        stopwords = set(pd.read_csv(filepath, header=None, quoting=3, encoding='utf-8')[0])
        print(f"✅ 停用词加载成功，数量：{len(stopwords)}")
    except Exception as e:
        print(f"⚠️ 停用词加载失败: {e}")
        stopwords = set()
    return stopwords

stop_words = load_stopwords(stopwords_path)

# 去除停用词和长度小于等于 1 的词语
tokenized_texts = [
    [word for word in doc if word not in stop_words and len(word) > 1]
    for doc in tokenized_texts
]
print(f"✅ 去除停用词后示例：{tokenized_texts[:5]}")

# 检查是否存在空文档
non_empty_texts = [doc for doc in tokenized_texts if doc]
if not non_empty_texts:
    print("❌ 预处理后文本为空，可能是数据过少或过滤条件过严。请检查原始数据和预处理步骤！")
    exit()

# 5. 创建词典和语料库
dictionary = corpora.Dictionary(non_empty_texts)
corpus = [dictionary.doc2bow(text) for text in non_empty_texts]
print(f"✅ 词典大小：{len(dictionary)}")
print(f"✅ 语料库样本数：{len(corpus)}")

# 检查词典和语料库是否有效
if len(dictionary) == 0 or len(corpus) == 0:
    print("❌ 字典或语料库为空，无法训练 LDA 模型。请检查数据预处理步骤！")
    exit()

# 6. 训练 LDA 模型
num_topics = 5  # 主题数，可根据需要调整
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=10
)
print("✅ LDA 模型训练完成！")

# 7. 输出每个主题的关键词
print("\n🎉 主题关键词：")
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"主题 {idx + 1}: {topic}")

# 8. 可视化
# 准备可视化数据
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# 保存为 HTML 文件
html_path = 'lda_visualization.html'
pyLDAvis.save_html(vis, html_path)
print(f"\n✅ LDA 可视化已保存为 {html_path}")

# 自动打开浏览器显示结果
webbrowser.open(html_path)

# 9. 每条评论的主题分布
doc_topics = []
for doc in corpus:
    topic_probs = lda_model.get_document_topics(doc, minimum_probability=0)
    doc_topics.append([prob for _, prob in topic_probs])

topic_df = pd.DataFrame(doc_topics, columns=[f'Topic {i + 1}' for i in range(num_topics)])
print("\n✅ 每条评论的主题分布 (前 5 行)：")
print(topic_df.head())

# 保存主题分布结果到 CSV
topic_df.to_csv('topic_distribution.csv', index=False)
print("✅ 每条评论的主题分布已保存为 topic_distribution.csv")

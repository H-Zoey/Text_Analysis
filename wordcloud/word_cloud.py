from wordcloud import WordCloud
import jieba
from collections import Counter
import matplotlib.pyplot as plt

#加载停用词
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    return stopwords

stopwords = load_stopwords("C:/Users/HP/TextAnalysis/wordcloud/stopwords-zh.txt")

font_path = 'msyh.ttc'

with open('E:/Python/TextAnalysis/Educated(Tara Westover).txt', 'r', encoding='utf-8') as file:
    chinese_novel_text = file.read()

#jieba 分词
seg_list = jieba.lcut(chinese_novel_text)

#去除停用词
filtered_seg_list = [word for word in seg_list if word not in stopwords]

#词频统计
word_counts = Counter(filtered_seg_list)
top_words = word_counts.most_common(10)
print("Top 10 高频词：")
for i, (word, freq) in enumerate(top_words, 1):
    print(f"{i}. {word}：{freq}")


wordcloud_text = ' '.join(filtered_seg_list)
wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(wordcloud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
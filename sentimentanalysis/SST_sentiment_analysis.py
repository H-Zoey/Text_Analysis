import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer
from transformers import TrainingArguments
from datasets import Dataset
import torch

indices = []
scores_list = []

with open('C:/Users/HP/TextAnalysis/sentimentanalysis/rawscores_exp12.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        index = int(parts[0])
        scores = list(map(int, parts[1:]))
        indices.append(index)
        scores_list.append(scores)

# 构造 DataFrame
raw_df = pd.DataFrame({'index': indices, 'scores': scores_list})
raw_df['avg_score'] = raw_df['scores'].apply(lambda x: sum(x)/len(x))

text_df = pd.read_csv('C:/Users/HP/TextAnalysis/sentimentanalysis/sentlex_exp12.txt', header=None, names=['index', 'phrase'])
text_df = text_df.dropna(subset='phrase')

# 合并 phrase 和 avg_score
df = pd.merge(text_df, raw_df[['index', 'avg_score']], on='index')

# 打情感分类标签（5档）
def score_to_class(score):
    if score <= 5:
        return 'very negative'
    elif score <= 10:
        return 'negative'
    elif score <= 15:
        return 'neutral'
    elif score <= 20:
        return 'positive'
    else:
        return 'very positive'

df['sentiment_class'] = df['avg_score'].apply(score_to_class)

# 最终数据
#print(df.head())

##可视化
#sns.countplot(x='sentiment_class', data=df, order=['very negative', 'negative', 'neutral', 'positive', 'very positive'])
#plt.title('Sentiment Class Distribution')
#plt.xticks(rotation=45)
#plt.show()

#标签转数字
label_mapping = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4
}
df['label'] = df['sentiment_class'].map(label_mapping)

#文本向量化
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X = tfidf.fit_transform(df['phrase'])
y = df['label']


#分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

##回归模型
#clf = LogisticRegression(max_iter=1000)
#clf.fit(X_train, y_train)
#clf_pred = clf.predict(X_test)
#
#print(f"Logistic Regression Classification Report:\n{classification_report(y_test, clf_pred, target_names=label.classes_)}")
#
##SVM
#svm_model = SVC(kernel='linear')
#svm_model.fit(X_train, y_train)
#svm_pred = svm_model.predict(X_test)
#
#print(f"SVM Classification Report:\n{classification_report(y_test, svm_pred, target_names=label.classes_)}")
#
#
##Random Forest
#rf_model = RandomForestClassifier(n_estimators=100, random_state=37)  # 100棵树的随机森林
#rf_model.fit(X_train, y_train)
#rf_pred = rf_model.predict(X_test)
#
#print(f"Random Forest Classification Report:\n{classification_report(y_test, rf_pred, target_names=label.classes_)}")

#BERT
id2label = {v: k for k, v in label_mapping.items()}
label2id = label_mapping

#减小数量加快测试速度
df = df.sample(n=3000, random_state=42).reset_index(drop=True)

#划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#将数据转换为 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

#加载 DistilBERT tokenizer 和模型
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# 定义 Tokenizer 函数
def tokenize_function(example):
    return tokenizer(example["phrase"], truncation=True, padding='max_length', max_length=128)

# 使用 map 函数将文本转换为 token
train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./distilbert_sentiment_output",     #模型保存路径
    learning_rate=2e-5,
    per_device_train_batch_size=16,                 #根据显卡性能调整
    per_device_eval_batch_size=16,
    num_train_epochs=3,                             #训练轮数
    eval_strategy="epoch",                    # 评估策略：每个epoch结束后评估
    save_strategy="epoch",                          # 保存策略：与evaluation_strategy一致
    load_best_model_at_end=True,                    # 训练完成后加载最佳模型
    logging_dir="./logs",                           # 日志目录
    logging_steps=50,                               # 每50步记录一次日志
)

# 创建 Trainer 对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    processing_class=tokenizer,
)

# 训练模型
trainer.train()

# 预测标签
predictions, label_ids, metrics = trainer.predict(test_tokenized)

# 计算分类报告
predicted_labels = predictions.argmax(axis=1)
print(classification_report(label_ids, predicted_labels, target_names=label_mapping.keys()))



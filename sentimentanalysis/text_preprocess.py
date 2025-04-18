import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt_tab')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(filtered_text)


text = "I am really happy to see you! But I am also a little sad that you have to leave."
processed_text = preprocess_text(text)
print("Processed Text:", processed_text)
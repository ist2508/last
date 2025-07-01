import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import os

stopwords = set(STOPWORDS)
stopwords.update([
    'https', 'co', 'RT', '...', 'amp', 'lu', 'deh', 'fyp', 'ya', 'gue',
    'sih', 'yg', 'nya', 'aja', 'sdh', 'gak', 'ga', 'nak', 'bae', 'min', 'kita'
])

def create_wordcloud(text, filename):
    wc = WordCloud(
        stopwords=stopwords,
        background_color="white",
        max_words=500,
        width=800,
        height=400
    ).generate(text)

    os.makedirs("hasil", exist_ok=True)
    wc.to_file(f"hasil/{filename}")

def plot_top_words(df, sentiment_value, filename):
    text = df[df['Predicted'] == sentiment_value]['steming_data'].str.cat(sep=' ')
    tokens = [word for word in text.split() if word not in stopwords]
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)

    if not top_words:
        return

    words, counts = zip(*top_words)
    colors = plt.cm.Pastel1(range(len(words)))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(words, counts, color=colors)
    plt.title(f"Top Words - Sentimen {sentiment_value}", fontsize=16)
    plt.xlabel("Kata", fontsize=12)
    plt.ylabel("Frekuensi", fontsize=12)
    plt.xticks(rotation=45)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(count), ha='center', va='bottom')

    os.makedirs("hasil", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"hasil/{filename}")
    plt.close()

def plot_sentiment_distribution(df):
    sentiment_counts = df['Predicted'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'orange', 'green'])
    ax.set_title("Distribusi Sentimen Prediksi Naive Bayes")
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah Tweet")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(height), ha='center', va='bottom')

    os.makedirs("hasil", exist_ok=True)
    fig.tight_layout()
    fig.savefig("hasil/sentimen_distribution.png")
    plt.close()

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Fungsi training dan evaluasi Multinomial Naive Bayes
def run_naive_bayes(labelled_file="Hasil_Labelling_Data.csv"):
    df = pd.read_csv(labelled_file)
    df.dropna(subset=['steming_data', 'Sentiment'], inplace=True)
    X = df['steming_data']
    y = df['Sentiment']

    # TF-IDF
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)

    # Evaluasi
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['Negatif', 'Netral', 'Positif'])
    class_report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Simpan confusion matrix sebagai gambar
    os.makedirs("hasil", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif', 'Netral', 'Positif'],
                yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.title('Confusion Matrix (MultinomialNB)')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    plt.savefig("hasil/conf_matrix_mnb.png")
    plt.close()

    # Simpan hasil prediksi
    result_df = pd.DataFrame({
        'steming_data': df.loc[y_test.index, 'steming_data'],
        'Actual': y_test,
        'Predicted': y_pred
    })
    result_df.to_csv("hasil/Hasil_pred_MultinomialNB.csv", index=False, encoding='utf8')

    return accuracy, class_report, conf_matrix, result_df, X_train.shape[0], X_test.shape[0]

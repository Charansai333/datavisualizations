import streamlit as st
import pdfplumber
import docx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np
import re

# 📄 Extract text from PDF using pdfplumber
def extract_text_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "").strip() + " "
    return text

# 📄 Extract text from DOCX
def extract_text_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# 🧹 Basic text preprocessing without NLTK
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

# ☁ Generate Word Cloud
def show_wordcloud(tokens):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(tokens))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# 📊 Scatter Plot of Word Frequencies
def show_scatter(tokens):
    freq = Counter(tokens).most_common(50)  # top 50 words
    words, counts = zip(*freq)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(len(words)), counts, c=counts, cmap="viridis", s=100, alpha=0.7)
    for i, word in enumerate(words):
        ax.text(i, counts[i] + 0.5, word, rotation=45, ha='right', fontsize=8)
    ax.set_title("Scatter Plot of Word Frequencies")
    ax.set_xlabel("Words (Top 50)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# 📉 Histogram of Word Frequencies
def show_histogram(tokens):
    freq = Counter(tokens)
    counts = list(freq.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(counts, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Word Frequencies")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Number of Words")
    st.pyplot(fig)

# 🔥 Heatmap of Word Co-occurrence
def show_heatmap(tokens, window_size=5):
    top_words = [word for word, _ in Counter(tokens).most_common(20)]
    matrix = np.zeros((20, 20))
    for i, word1 in enumerate(top_words):
        for j, word2 in enumerate(top_words):
            count = sum(1 for k in range(len(tokens) - window_size)
                        if word1 in tokens[k:k+window_size] and word2 in tokens[k:k+window_size])
            matrix[i][j] = count
    df = pd.DataFrame(matrix, index=top_words, columns=top_words)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, cmap='YlGnBu', annot=True, ax=ax)
    st.pyplot(fig)

# 🚀 Streamlit UI
st.title("📚 Text Visualization App")
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        raw_text = extract_text_pdf(uploaded_file)
    else:
        raw_text = extract_text_docx(uploaded_file)

    st.subheader("Extracted Text Preview")
    st.write(raw_text[:1000] + "...")  # Show preview

    tokens = preprocess_text(raw_text)

    if not tokens:
        st.warning("No text found in the uploaded document.")
    else:
        option = st.selectbox("Choose Visualization", ["Word Cloud", "Scatter Plot", "Histogram", "Heatmap"])

        if option == "Word Cloud":
            show_wordcloud(tokens)
        elif option == "Scatter Plot":
            show_scatter(tokens)
        elif option == "Histogram":
            show_histogram(tokens)
        elif option == "Heatmap":
            show_heatmap(tokens)

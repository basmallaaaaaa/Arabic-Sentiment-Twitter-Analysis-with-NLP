import nltk
nltk.download('stopwords')

import streamlit as st
import re
import pickle
import emoji
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -------------------
# Load model + tokenizer
# -------------------
model = load_model("model_BI.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------
# Preprocessing functions
# -------------------
pun = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
stopwords_list = stopwords.words('arabic') + stopwords.words('english')

def remove_tags(text):
    return re.sub(r'<.*?>', ' ', text)

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)

def remove_emoji(text):
    return emoji.demojize(text)

def remove_digits(text):
    return re.sub(r'\d+', ' ', text)

def remove_pun(text):
    for char in pun:
        text = text.replace(char, '')
    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords_list])

def preprocess(text):
    text = remove_tags(text)
    text = remove_urls(text)
    text = remove_emoji(text)
    text = remove_digits(text)
    text = remove_pun(text)
    text = remove_stopwords(text)
    return text

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Arabic Sentiment Analysis", page_icon="✨")

st.title("✨ Arabic Twitter Sentiment Analysis")
st.write("🚀 BiLSTM model to classify tweets into Positive / Negative sentiment")

user_input = st.text_area("📝 Enter your tweet:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a tweet first")
    else:
        # Preprocess
        clean_text = preprocess(user_input)

        # Tokenize
        seq = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=100)  # use same maxlen from training

        # Prediction
        pred = model.predict(padded)[0][0]

        # Output
        if pred >= 0.5:
            st.success(f"✅ Positive 😀 (score: {pred:.4f})")
        else:
            st.error(f"❌ Negative 😞 (score: {pred:.4f})")

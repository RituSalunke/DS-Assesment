import streamlit as st
import joblib
import re

# Load saved model and vectorizer
model = joblib.load(r"D:\nlp_assesment\news_category_model.pkl")
vectorizer = joblib.load(r"D:\nlp_assesment\count_vectorizer.pkl")

# Text preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="News Category Predictor", page_icon="📰")

st.title("News Category Prediction App")
st.write("Enter a news headline or short article to predict its category.")

user_input = st.text_area("Enter News Text:", height=150)

if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("please enter some text")
    else:
        clean_text = preprocess_text(user_input)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]

        st.success(f" Predicted Category: **{prediction}**")

st.markdown("---")
st.caption("Built using NLP, Logistic Regression & Streamlit")
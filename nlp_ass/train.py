import pandas as pd
import streamlit as st
import joblib

import spacy
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_json(News_Category_Dataset_v3.json",lines=True)
st.dataframe(df.head(20))

keep_categories = ["TECH", "ENTERTAINMENT", "POLITICS", "BUSINESS"]

filtered_df = df[df["category"].isin(keep_categories)]

filtered_df.head()

filtered_df["category"].value_counts()

#filtered_df.to_csv("news_filtered.csv", index=False)

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

#nlp = spacy.blank("en")

#3) Write a function to preprocess headlines: lowercase, removestopwords, punctuation, and lemmatize.

def preprocess_headline(text):
    doc = nlp(text)

    cleaned_tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return " ".join(cleaned_tokens)

headline = "Apple Launches New AI-Powered Devices in 2025!"

print(preprocess_headline(headline))

#4) Apply preprocessing to all headlines.

# Apply preprocessing
filtered_df["clean_headline"] = filtered_df["headline"].apply(preprocess_headline)

# View result in table format
filtered_df[["headline", "clean_headline"]].head()

#5) Convert text into numeric vectors using CountVectorizer with unigrams and bigrams.

#6) Limit the vocabulary size to 5000 features; explain why.

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=5000    # limit vocabulary size
)

#7) Create feature matrix X and label vector y.

# Feature matrix (already created using CountVectorizer)
X = vectorizer.fit_transform(filtered_df["clean_headline"])

# Save vectorizer
joblib.dump(vectorizer,count_vectorizer.pkl")
st.write("save vectorizer model")

y = filtered_df["category"]

print("X shape:", X.shape)
print("y shape:", y.shape)

#8) Split data into training and test sets with balanced categories

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,               # feature matrix
    y,               # labels
    test_size=0.2,   # 20% test data
    random_state=42,
    stratify=y       # IMPORTANT: keeps category balance
)

print("Train category distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest category distribution:")
print(y_test.value_counts(normalize=True))

#9) Train Logistic Regression model with enough iterations.

from sklearn.linear_model import LogisticRegression

# Initialize model with enough iterations
log_reg = LogisticRegression(
    max_iter=1000,      # enough iterations for convergence
    solver="lbfgs",
    n_jobs=-1
)

# Train the model
log_reg.fit(X_train, y_train)

joblib.dump(log_reg, news_category_model.pkl")
st.write("save logistic regression model")

#10)Test the model and report accuracy

from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = log_reg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)

#11)Build a function to predict category for new headlines

def predict_category(headline):
    # 1. Preprocess the headline
    clean_text = preprocess_headline(headline)

    # 2. Vectorize the text
    text_vector = vectorizer.transform([clean_text])

    # 3. Predict category

    prediction = log_reg.predict(text_vector)

    return prediction[0]

print(predict_category("Apple launches new AI powered iPhone"))

print(predict_category("Government passes new economic reform bill"))






















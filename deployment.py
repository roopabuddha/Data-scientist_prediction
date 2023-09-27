

import streamlit as st

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import re
import string
import nltk
import  spacy

with open("svm_model (1).pkl", "rb") as file:
    model = pickle.load(prediction_code.py)

with open("tfidf_vectorizer (1).pkl", "rb") as file:
    vectorizer = pickle.load(prediction_code.py)

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = text.lower()
    return text.strip()

def remove_punctuation(text):
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    return punctuation_free

def tokenization(text):
    tokens = re.split(' ', text)
    return tokens

def remove_stopwords(text):
    output = " ".join(i for i in text if i not in stopwords)
    return output

def lemmatizer(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if not token.text in set(stopwords)]
    return ' '.join(sent)

st.title("Data Scientist Prediction App")
st.markdown("By rupa srija sirisha priya")
image = Image.open("data prediction.png")
st.image(image, use_column_width=True)

st.title("Data Scientist Salary Prediction")

user_input = st.text_area("Enter the job description:")

if user_input:
    user_input = clean_text(user_input)
    user_input = remove_punctuation(user_input)
    user_input = user_input.lower()
    user_input = tokenization(user_input)
    user_input = remove_stopwords(user_input)
    user_input = lemmatizer(user_input)

if st.button("Predict Salary"):
    if user_input:
        text_vectorized = vectorizer.transform([user_input])

        predicted_salary = model.predict(text_vectorized)[0]

        st.header("Predicted Salary:")
        st.subheader(f"The predicted salary for the given job description is: ${predicted_salary:,.2f}")
    else:
        st.subheader("Please enter a job description for salary prediction.")


import streamlit as st
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        ## remove special characters
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    ## Remove stop words and punctuation
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

## Stemming
    for i in text:
        y.append(ps.stem(i))

    str_y = " ".join(y)

    return str_y


## When input text comes in, you first preprocess it, then vectorize, then pass it to model to predict

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if(st.button("Predict")):

    transformed_sms = transform_text(input_sms)

    # st.write(transformed_sms)

    vector_input  = tfidf.transform([transformed_sms])

    # st.write(vector_input)

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


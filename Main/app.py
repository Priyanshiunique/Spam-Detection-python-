import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("SMS Alert Filter")






content = """
**Welcome to the SMS Alert Filter**!

This tool helps you identify and filter out spam messages from your inbox. Simply enter any SMS text, and our model will classify it as either Spam or Not Spam. It uses advanced machine learning techniques to accurately analyze and detect unwanted messages, helping you maintain a cleaner and safer messaging experience.

Try it now!
"""
st.write(content)
    

# input_sms = st.text_input("Enter the SMS")
input_sms = st.text_area("Enter the SMS", height=150, placeholder="Type your SMS message here...")


if st.button('Verify SMS'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tk.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("SMS is spam!")
    else:
        st.header("SMS is not Spam.")

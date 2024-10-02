import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the saved vectorizer and Naive Bayes model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# transform_text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    # Remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    # Stem the words
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return ' '.join(y)


# streamlit code
st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    #Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict
    result = model.predict(vector_input) [0]
    
    #Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
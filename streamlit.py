import streamlit as st
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

st.title("Document Classifier")
st.write("Upload a text file to classify its genre into one of the following categories: business, world, sports, sci/tech")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    
    st.write("File Content:")
    st.write(content)
    
    response = requests.post("http://backend:5000/predict", json={"text": content})
    if response.status_code == 200:
        result = response.json()
        st.write("Predicted Genre:")
        st.write(f"**{result['prediction']}**")
    else:
        st.write("Error: Could not get prediction")

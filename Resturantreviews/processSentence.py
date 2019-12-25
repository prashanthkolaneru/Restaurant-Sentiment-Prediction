import pandas as pd
import re
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pickle

vec = pickle.load(open('Resturant_t.pkl','rb'))


tfidf_vt = TfidfVectorizer(lowercase=False,vocabulary=None,tokenizer=None)

def process_features(sentences):    
    Corpus = []
    for i in range(0,1000):
        review = re.sub('[^a-zA-Z]', ' ', sentences[i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
        review = " ".join(review)
        Corpus.append(review)
    X_vt = vec.transform(Corpus)
    return X_vt
                             




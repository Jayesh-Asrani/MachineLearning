import string

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import random

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC

wordnet_lemmatizer = WordNetLemmatizer()

positive_reviews = open('positive_review.txt', "r", encoding='latin-1').read()
negative_reviews = open('negative_review.txt', "r", encoding='latin-1').read()

documents = []
all_words = []

for review in positive_reviews.split('\n'):
    documents.append((review, "pos"))

for review in negative_reviews.split('\n'):
    documents.append((review, "neg"))

DataSet = pd.DataFrame(documents)
DataSet.columns = ['Text', 'Label']

cleaned_text = []

for sentence in DataSet.Text:
    words = nltk.tokenize.word_tokenize(sentence)
    clean_words = set()
    for word in words:
        word = word.lower()
        if word not in set(stopwords.words('english')) and word not in string.punctuation:
            clean_words.add(word)
        cleaned_text.append(list(clean_words))

DataSet['Cleaned_Text'] = cleaned_text


le = LabelEncoder()
cv = TfidfVectorizer()
X = cv.fit_transform(DataSet['Cleaned_Text']).toarray()
y = DataSet.iloc[:, 0].values
y = le.fit_transform(y)

DataSet.dropna()
DataSet.reindex
random.shuffle(DataSet)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
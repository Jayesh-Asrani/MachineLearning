import string

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.cross_validation import train_test_split
import random
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

for review in positive_reviews.split('\n'):
    documents.append((review, "pos"))

for review in negative_reviews.split('\n'):
    documents.append((review, "neg"))

DataSet = pd.DataFrame(documents)
DataSet.columns = ['Text', 'Label']

cleaned_text = []

for sentence in DataSet.Text:
    words = nltk.tokenize.word_tokenize(sentence)
    pos = nltk.pos_tag(words)
    clean_words = set()
    for word in words:
        word = word.lower()
        if word not in set(stopwords.words('english')) and word not in string.punctuation and pos[words.index(word)][
            0] in ["J"]:
            clean_words.add(word)
    cleaned_text.append(" ".join(list(clean_words)))

DataSet['Cleaned_Text'] = cleaned_text

DataSet.dropna()
DataSet.reindex

le = LabelEncoder()
cv = TfidfVectorizer()
X = cv.fit_transform(DataSet['Cleaned_Text']).toarray()
y = DataSet['Label'].values
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

MultinomialNB_classifier = MultinomialNB()
MultinomialNB_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = MultinomialNB_classifier.predict(X_test)

print("MultinomialNB_classifier Test Score : ", MultinomialNB_classifier.score(X_test, y_test))

BernoulliNB_classifier = BernoulliNB()
BernoulliNB_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = BernoulliNB_classifier.predict(X_test)

print("BernoulliNB_classifier Test Score : ", BernoulliNB_classifier.score(X_test, y_test))

SGDC_Classifier = SGDClassifier()
SGDC_Classifier.fit(X_train, y_train)

y_pred = SGDC_Classifier.predict(X_test)

print("SGDC_Classifier Test Score : ", SGDC_Classifier.score(X_test, y_test))

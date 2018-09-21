# NLP

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# Importing DataSet
Reviews_DataSet = pd.read_csv('Restaurant_Reviews.tsv', sep='\t', quoting=3)
PS_Stem = PorterStemmer()
stop_words = stopwords.words("english")
corpus = []

# Text Cleaning
for review in Reviews_DataSet.Review:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [PS_Stem.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    corpus.append(review)

# Creating Bag of Words Model
CV = CountVectorizer(max_features=1500)
X = CV.fit_transform(corpus).toarray()
Y = Reviews_DataSet.iloc[:, 1]

X_train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25)

NB_Multinomial = MultinomialNB()
NB_Multinomial.fit(X_train, Y_Train)

Y_Pred = NB_Multinomial.predict(X_Test)
cm = confusion_matrix(Y_Test, Y_Pred)

# TF IDF Vectorizer

TF = TfidfVectorizer(max_features=1500)
X = TF.fit_transform(corpus).toarray()
Y = Reviews_DataSet.iloc[:, 1]

X_train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25)

SVC_Model = SVC(kernel='linear',random_state=0)
SVC_Model.fit(X_train, Y_Train)

Y_Pred = SVC_Model.predict(X_Test)
cm = confusion_matrix(Y_Test, Y_Pred)

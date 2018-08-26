#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:53:47 2018

@author: fractaluser
"""

import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

wordnet_lemmatizer = WordNetLemmatizer()

stop_words = set(w.rstrip() for w in open('stopwords.txt'))

posistive_reviews = BeautifulSoup(open('electronics/positive.review').read())
posistive_reviews = posistive_reviews.find_all('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.find_all('review_text')

np.random.shuffle(posistive_reviews)
posistive_reviews = posistive_reviews[:len(negative_reviews)]


def my_tokenizer(s):
    if s is not None:
        s = s.lower()
        tokens = word_tokenize(s)
        tokens = [t for t in tokens if len(t) > 2]
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t not in stop_words]
        return tokens


word_map_index = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

for review in posistive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_map_index:
            word_map_index[token] = current_index
            current_index = current_index + 1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_map_index:
            word_map_index[token] = current_index
            current_index = current_index + 1


def token_to_vectors(tokens, lable):
    x = np.zeros(len(word_map_index) + 1)
    for token in tokens:
        i = word_map_index[token]
        x[i] += 1
    x = x / x.sum()
    x[-1] = lable
    return x


N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_map_index) + 1))
i = 0

for tokens in positive_tokenized:
    xy = token_to_vectors(tokens, 1)
    data[i, :] = xy
    i += 1

for tokens in negative_tokenized:
    xy = token_to_vectors(tokens, 0)
    data[i, :] = xy
    i += 1

np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]

Xtest = X[-100:, ]
Ytest = Y[-100:, ]

logistic_model = LogisticRegression()
logistic_model.fit(Xtrain, Ytrain)
print("Classification Rate", logistic_model.score(Xtest, Ytest))


logistic_model = LogisticRegression()
logistic_model.fit(Xtrain, Ytrain)
print("Classification Rate", logistic_model.score(Xtest, Ytest))

multinomial_model = MultinomialNB()
multinomial_model.fit(Xtrain, Ytrain)
print("Classification Rate", multinomial_model.score(Xtest, Ytest))
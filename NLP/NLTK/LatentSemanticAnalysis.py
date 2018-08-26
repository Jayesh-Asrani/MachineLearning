#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 11:56:49 2018

@author: fractaluser
"""

import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

wordnet_lemmatizer = WordNetLemmatizer()

titles = [w.rstrip() for w in open('all_book_titles.txt')]
stop_words = set(w.rstrip() for w in open('stopwords.txt'))

stop_words = stop_words.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth', })


def my_tokenizer(s):
    s = s.lower()  # downcase
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]  # remove any digits, i.e. "3rd edition"
    return tokens


word_map_index = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []

for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_map_index:
                word_map_index[token] = current_index
                current_index = current_index + 1
                index_word_map.append(token)
    except:
        pass


def tokens_to_vector(tokens):
    x = np.zeros(len(word_map_index))
    for t in tokens:
        i = word_map_index[t]
        x[i] = 1
    return x


N = len(all_tokens)
D = len(word_map_index)

X = np.zeros((D, N))
i = 0
for tokens in all_tokens:
    X[:, i] = tokens_to_vector(tokens)
    i += 1

svd = TruncatedSVD()
Z = svd.fit_transform(X)

plt.scatter(Z[:, 0], Z[:, 1])
for i in range(D):
    plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))
plt.show()

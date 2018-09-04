#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:59:52 2018

@author: fractaluser
"""

# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Loading Data
Startups_DataSet = pd.read_csv('50_Startups.csv')

X = Startups_DataSet.iloc[:, :-1].values
Y = Startups_DataSet.iloc[:, 4].values

# Encoding Categorical Variable
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Dropping one Dummy Variable to avoid Dummy variable trap
X = X[:, 1:]

# Splitting DataSet
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting the Linear model

ML_Regression = LinearRegression()
ML_Regression.fit(X_train, Y_train)

# Prediting the Results

Y_pred = ML_Regression.predict(X_test)

# Building Backward Elimination
X=np.append(arr=np.ones((50,1)),values=X,axis=1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:02:22 2018

@author: Jayesh Asrani
"""
# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Loading Data
Salary_DataSet = pd.read_csv('Salary_Data.csv')

X = Salary_DataSet.iloc[:, :-1].values
Y = Salary_DataSet.iloc[:, 1].values

# Splitting DataSet
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# Fitting Linear Regression
LR_Model = LinearRegression()
LR_Model.fit(X_train, Y_train)

# Predicting Values
Y_pred = LR_Model.predict(X_test)

# Visualising the results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, LR_Model.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, LR_Model.predict(X_train), color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()

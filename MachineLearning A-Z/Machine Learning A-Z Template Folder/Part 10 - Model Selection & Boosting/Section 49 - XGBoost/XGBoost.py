# XGBoost

# Importing Libraries

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

Churn_DataSet = pd.read_csv('Churn_Modelling.csv')
X = Churn_DataSet.iloc[:, 3:13].values
Y = Churn_DataSet.iloc[:, 13].values

# Encoding the Categorical Variables
LE_x_1 = LabelEncoder()
X[:, 1] = LE_x_1.fit_transform(X[:, 1])
LE_x_2 = LabelEncoder()
X[:, 2] = LE_x_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the DataSet
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting XGBoost

XGB_Model = XGBClassifier(n_estimators=200)
XGB_Model.fit(X_Train, Y_Train)

# Predict
Y_Pred = XGB_Model.predict(X_Test)

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

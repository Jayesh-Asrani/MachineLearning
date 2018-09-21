# Artifical Neural Network

# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

Bank_DataSet = pd.read_csv('Churn_Modelling.csv')

X = Bank_DataSet.iloc[:, 3:13].values
Y = Bank_DataSet.iloc[:, 13].values

# Encoding the Categorical features
LE_Sex = LabelEncoder()
X[:, 1] = LE_Sex.fit_transform(X[:, 1])

LE_City = LabelEncoder()
X[:, 2] = LE_City.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy trap
X = X[:, 1:]

# Train Data split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Feature Scaling

SC_X = StandardScaler()
X_Train = SC_X.fit_transform(X_Train)
X_Test = SC_X.fit_transform(X_Test)

# ANN Model

ANN_Model = Sequential()

ANN_Model.add(Dense(input_dim=11, activation='relu', kernel_initializer='uniform', units=8))
ANN_Model.add(Dense(activation='relu', kernel_initializer='uniform', units=8))
ANN_Model.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

ANN_Model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the ANN

history = ANN_Model.fit(X_Train,
                        Y_Train,
                        epochs=75,
                        batch_size=20,
                        verbose=1)
# Predict Probabilities

Y_Pred = ANN_Model.predict(X_Test)
Y_Pred = (Y_Pred > 0.5)

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

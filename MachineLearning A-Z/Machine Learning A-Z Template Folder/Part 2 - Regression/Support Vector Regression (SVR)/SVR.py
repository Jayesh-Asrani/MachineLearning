# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Loading Data
Salary_DataSet = pd.read_csv('Position_Salaries.csv')

X = Salary_DataSet.iloc[:, 1:-1].values
Y = Salary_DataSet.iloc[:, 2].values.reshape(10, 1)

# Feature Scaling
std_scaler_X = StandardScaler()
X = std_scaler_X.fit_transform(X)

std_scaler_Y = StandardScaler()
Y = std_scaler_Y.fit_transform(Y)

# Fitting the SVR model

SVR_Model = SVR(kernel='rbf')
SVR_Model.fit(X, Y)

# Prediting the Results
y_pred = std_scaler_Y.inverse_transform(SVR_Model.predict(std_scaler_X.transform(np.array([[6.5]]))))

# Visualising the results
plt.scatter(X, Y, color='Red')
plt.plot(X, SVR_Model.predict(X), color='Blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Salary vs Position')

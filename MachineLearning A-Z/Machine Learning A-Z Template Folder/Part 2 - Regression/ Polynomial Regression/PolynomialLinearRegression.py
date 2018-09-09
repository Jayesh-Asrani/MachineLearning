import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm

# Loading Data
Salary_DataSet = pd.read_csv('Position_Salaries.csv')

X = Salary_DataSet.iloc[:, 1:-1].values
Y = Salary_DataSet.iloc[:, 2].values

# Fitting Linear Regression

LR_Regression = LinearRegression()
LR_Regression.fit(X, Y)

# Fitting Polynomial Regression

PL_Features = PolynomialFeatures(degree=3)
X_Poly = PL_Features.fit_transform(X)

LR_Regression_2 = LinearRegression()
LR_Regression_2.fit(X_Poly, Y)

# Visualising Linear Model results
plt.scatter(X, Y, color='Red')
plt.plot(X, LR_Regression.predict(X), color='Blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Truth or Bluff')

# Visualising Polynomial Model results with degree 2
plt.scatter(X, Y)
plt.plot(X, LR_Regression_2.predict(PL_Features.fit_transform(X)), color='Blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Truth or Bluff')

# Visualising Polynomial Model results with degree 3 step size .1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, Y)
plt.plot(X_grid, LR_Regression_2.predict(PL_Features.fit_transform(X_grid)), color='Blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Truth or Bluff')


# Predicting a value with Linear model
LR_Regression.predict(6.5)

# Predicting a value with Polynomial Model
LR_Regression_2.predict(PL_Features.fit_transform(6.5))
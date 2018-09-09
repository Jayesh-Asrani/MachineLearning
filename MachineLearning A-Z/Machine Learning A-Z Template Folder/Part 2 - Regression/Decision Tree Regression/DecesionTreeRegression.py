# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Loading Data
Salary_DataSet = pd.read_csv('Position_Salaries.csv')

X = Salary_DataSet.iloc[:, 1:-1].values
Y = Salary_DataSet.iloc[:, 2].values.reshape(10, 1)

# Fitting the Decision Tree model

DTR_Model = DecisionTreeRegressor(random_state=0)
DTR_Model.fit(X, Y)

# Prediting the Results
y_pred = DTR_Model.predict(6.5)

# Visualising the results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, Y, color='Red')
plt.plot(X_grid, DTR_Model.predict(X_grid), color='Blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Salary vs Position')

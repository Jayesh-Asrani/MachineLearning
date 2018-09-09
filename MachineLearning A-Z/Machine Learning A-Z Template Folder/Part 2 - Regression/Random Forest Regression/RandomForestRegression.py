# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Loading Data
Salary_DataSet = pd.read_csv('Position_Salaries.csv')

X = Salary_DataSet.iloc[:, 1:-1].values
Y = Salary_DataSet.iloc[:, 2].values.reshape(10, 1)

# Fitting the Decision Tree model

RF_Model = RandomForestRegressor(random_state=0,n_estimators=300)
RF_Model.fit(X, Y)

# Prediting the Results
y_pred = RF_Model.predict(6.5)

# Visualising the results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, Y, color='Red')
plt.plot(X_grid, RF_Model.predict(X_grid), color='Blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Salary vs Position')

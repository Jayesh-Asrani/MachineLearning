# SVM Classification With Kernel with Grid Search

# Importing Libraries
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

Social_DataSet = pd.read_csv('Social_Network_Ads.csv')

X = Social_DataSet.iloc[:, [2, 3]].values
Y = Social_DataSet.iloc[:, 4].values

# Train Data split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling

SC_X = StandardScaler()
X_Train = SC_X.fit_transform(X_Train)
X_Test = SC_X.fit_transform(X_Test)

# Fitting SVC
SVC_Model = SVC(kernel='rbf', random_state=0)
SVC_Model.fit(X_Train, Y_Train)

# Predicting the result sets
Y_Pred = SVC_Model.predict(X_Test)

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Applying K-Fold Cross Validation
accuracies = cross_val_score(estimator=SVC_Model, X=X_Train, y=Y_Train, cv=10)
accuracies.mean()
accuracies.std()

# Applying grid search to find best parameters

parameteres = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
               {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01]}
               ]
GridSearch = GridSearchCV(param_grid=parameteres, estimator=SVC_Model, scoring='accuracy', cv=10)
GridSearch.fit(X_Train, Y_Train)

# Visualising the results
X_set, y_set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, SVC_Model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVC (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

X_set, y_set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, SVC_Model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVC (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# PCA

# Importing Libraries

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

Wine_DataSet = pd.read_csv(
    '/home/fractaluser/MLGitHub Repo/MachineLearning A-Z/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Wine.csv')
X = Wine_DataSet.iloc[:, 0:13].values
Y = Wine_DataSet.iloc[:, -1].values

# Splitting Train and Test Set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Scaling the features

Std_Scaler = StandardScaler()
X_Train = Std_Scaler.fit_transform(X_Train)
X_Test = Std_Scaler.transform(X_Test)

# Applying PCA
PCA_Model = PCA(n_components=2)
X_Train = PCA_Model.fit_transform(X_Train)
X_Test = PCA_Model.transform(X_Test)
explained_variance = PCA_Model.explained_variance_

# Fitting the Logistic Model
LR_Model = LogisticRegression(random_state=0)
LR_Model.fit(X_Train, Y_Train)

# Prediting the results

Y_Pred = LR_Model.predict(X_Test)

# Confusion Matrix
CM = confusion_matrix(Y_Test, Y_Pred)
print(CM)

# Visualising the results
X_set, y_set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, LR_Model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression with PCA (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

X_set, y_set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, LR_Model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression with PCA (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

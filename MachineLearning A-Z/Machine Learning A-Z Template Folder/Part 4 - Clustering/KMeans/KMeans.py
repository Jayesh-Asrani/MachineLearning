# KMeans CLustering

# Importing Libraries
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Importing the DataSet
MallCust_DataSet = pd.read_csv('Mall_Customers.csv')
X = MallCust_DataSet.iloc[:, [3, 4]].values

# Using Elbow method to find K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting the KMeans 
KM_Model = KMeans(n_clusters=5, init='k-means++', random_state=42)
Y_kMeans = KM_Model.fit_predict(X)

# Plotting the Clusters
plt.scatter(X[Y_kMeans == 0, 0], X[Y_kMeans == 0, 1], s=100, color='red', label='Cluster 1')
plt.scatter(X[Y_kMeans == 1, 0], X[Y_kMeans == 1, 1], s=100, color='blue', label='Cluster 2')
plt.scatter(X[Y_kMeans == 2, 0], X[Y_kMeans == 2, 1], s=100, color='green', label='Cluster 3')
plt.scatter(X[Y_kMeans == 3, 0], X[Y_kMeans == 3, 1], s=100, color='cyan', label='Cluster 4')
plt.scatter(X[Y_kMeans == 4, 0], X[Y_kMeans == 4, 1], s=100, color='magenta', label='Cluster 5')
plt.scatter(KM_Model.cluster_centers_[:, 0], KM_Model.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
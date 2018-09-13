# Hierarchical CLustering

# Importing Libraries
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import hierarchical
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Importing the DataSet
MallCust_DataSet = pd.read_csv('Mall_Customers.csv')
X = MallCust_DataSet.iloc[:, [3, 4]].values

# Plotting Dendogram to find number of clusters
Dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()

# Fitting the Hierarchial Clustering
HC_Model = AgglomerativeClustering(n_clusters=5, linkage="ward")
Y_HC = HC_Model.fit_predict(X)

# Plotting the Clusters
plt.scatter(X[Y_HC == 0, 0], X[Y_HC == 0, 1], s=100, color='red', label='Cluster 1')
plt.scatter(X[Y_HC == 1, 0], X[Y_HC == 1, 1], s=100, color='blue', label='Cluster 2')
plt.scatter(X[Y_HC == 2, 0], X[Y_HC == 2, 1], s=100, color='green', label='Cluster 3')
plt.scatter(X[Y_HC == 3, 0], X[Y_HC == 3, 1], s=100, color='cyan', label='Cluster 4')
plt.scatter(X[Y_HC == 4, 0], X[Y_HC == 4, 1], s=100, color='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

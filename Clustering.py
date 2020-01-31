# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:33:32 2019

@author: chodiss
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
mc_df=pd.read_csv("Mall_Customers.csv")
# Print the Data Frame
#print(mc_df)
#print(mc_df["Spending Score (1-100)"].max())
# Check if there are any null values in the dataset
print(mc_df.isnull().sum())
# check the datatypes
print(mc_df.dtypes)


# Save the data features 'Annual Income (k$)','Spending Score (1-100)' to X

X=mc_df.loc[:,['Annual Income (k$)','Spending Score (1-100)']].values
#plt.scatter(X[:, 0], X[:, 1], s = 30, color ='b') 
#plt.xlabel('X') 
#plt.ylabel('Y') 
  
#plt.show() 
#X=mc_df.iloc[:,[3,4]].values

#print(X["Spending Score (1-100)"].max())
#print(X)
# Elbow Method
ks=range(1,11)
inertias=[]
for k in ks:
    model=KMeans(n_clusters=k)
    model.fit(X)
    inertias.append(model.inertia_)


plt.figure()
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.title('Elbow Method to find K',fontsize=20)
plt.xlabel('Number of Clusters',fontsize=20)
plt.ylabel('Inertia',fontsize=20)
#plt.text(3.5,inertias[4]+200,'K=5',color='red',fontsize=19)
plt.xticks(ks)
plt.show()

# From the plot we can pick k as 5 and fit
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
#  Predicting the clustering results y for data set X 
y_kmeans = kmeans.predict(X)
#print(y_kmeans)

# Visual representation of clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c ='orchid', label ='Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c ='Plum', label ='Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c ='Purple', label ='Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c ='Magenta', label ='Cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c ='Mediumpurple', label ='Cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 50, c ='red', label ='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
    
import scipy.cluster.hierarchy as sch 
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.show()

kmeans.fit(X)

#  Predicting the clustering results y for data set X 
y_kmeans = kmeans.predict(X)

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
#temp=hc.fit(X)
y_hc = hc.fit_predict(X)

#print(y_hc)

# visualising the clusters

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c ='#63C5DA', label ='Cluster1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c ='Teal', label ='Cluster2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c ='#0492C2', label ='Cluster3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 50, c ='#016064', label ='Cluster4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 50, c ='#52B2BF', label ='Cluster5')


plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#  Please describe the data matrix at the bottom of yoursource file.  Mark these as comments.
## How many observations are in this data set ?
### We can also use mc_df.count() which counts the values without NaN 

print("--------------------------------------------------")
print("Number of observations in the data set:",len(mc_df))
print("--------------------------------------------------")


# How many clusters you got by using K-Means ?  
print("5 Clusters using K-Means")

# How many clusters you got by using hierarchical clustering ?  
print("5 Clusters using hierarchial clustering")

# How you pick the number of clusters ?
print("By using the elbow and dendrogram where we have optimal clusters")
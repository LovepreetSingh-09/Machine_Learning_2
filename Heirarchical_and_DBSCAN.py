# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 22:05:26 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hierarchical clustering is of 2 types - Aglomerative and divisive
# Agglomerative clustering starts with one point as a single cluster to one complete cluster of all oints while divisive works other way around.
# agglomerative can be done by single linkage where the distance between the most similar or most closest points in a pair of clusters is measured to join clusters.
# In complete linkage, the most dissimilar points or the farthest points of two cluster is measured to join the two clusters.
# In average linkage, the average distance b/w all the points of 2 clusters is measured.
# In Ward's linkage, the clusters that leads to the minimum increase in the total within SSE is joined.
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0','ID_1','ID_2','ID_3','ID_4']
X = np.random.random_sample([5,3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

# Now calculate the distance matrix 
from scipy.spatial.distance import pdist,squareform
row_dist=pd.DataFrame(squareform(pdist(df,metric='euclidean')),columns=labels,index=labels)
print(row_dist)
# 10 values of distance b/w every possible pair of data frame
print(pdist(df,metric='euclidean'))
# square form makes a square matrix of index where each row represent the distance b/w the index row and all other index (columns)

from scipy.cluster.hierarchy import linkage
help(linkage)
# Here the default method is single
# The linkage class takes a flat array of distances that is returned by pdist which is also known as condensed matrix
row_clusters=linkage(pdist(df,metric='euclidean'),method='complete')
print(row_clusters)
# We can also give input to linkage as the raw values of the dataframe.
# The result is gonna be same
row_clusters=linkage(df.values,metric='euclidean',method='complete')
print(row_clusters)

cluster_df=pd.DataFrame(row_clusters,columns=['row label 1','row label 2','distance', 'no. of items in clust.'],
                        index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])
print(cluster_df)

# Visualizing clustering process by dendrogram
from scipy.cluster.hierarchy import dendrogram
row_dendr=dendrogram(row_clusters,labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean Distance')
plt.show()

# Attaching dendrogram with a heatmap
fig = plt.figure(figsize=(8,8), facecolor='white')
# Giving x-axis position, y-axis position, width and height
axd=fig.add_axes([0.09,0.1,0.2,0.6])
row_dendr=dendrogram(row_clusters,orientation='left')

# Getting the cluster no. getting attached in the descending order from dendrogram 
print(row_dendr['leaves'][::-1])
# reorder the initial data frame on the basis of clustering labels
df_rowcluster=df.iloc[row_dendr['leaves'][::-1]]
print(df_rowcluster)

axm = fig.add_axes([0.23,0.1,0.6,0.6])
cax = axm.matshow(df_rowcluster,interpolation='nearest', cmap='hot_r')
axd.set_xticks([])
axd.set_yticks([])
# Spine is the covering or border area of the dendrogram
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowcluster.columns))
axm.set_yticklabels([''] + list(df_rowcluster.index))
plt.show()

# Agglomerative CLustering in Scikit-Learn
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
labels=ac.fit_predict(X)
print(labels) # [1 0 0 2 1]
# The labels after clusteing is same as we did earlier


# Density-based Spatial Clustering of Applications with Noise (DBSCAN) :-
# A point is considered a core point if at least a specified number (MinPts) of neighboring points fall within the specified radius
# A border point is a point that has fewer neighbors than MinPts within, but lies within the radius of a core point
# All other points that are neither core nor border points are considered noise points

# Cluster teo half moon shape 
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200,noise=0.05, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()
# KMeans and Agglomerative are unable cluster this shape
# So now try DBSCAN
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=0.2,min_samples=5,metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db==0,0],X[y_db==0,1], c='lightblue',edgecolor='black',
            marker='o', s=40,label='cluster 1')
plt.scatter(X[y_db==1,0], X[y_db==1,1],c='red',edgecolor='black',
            marker='s', s=40,label='cluster 2')
plt.legend()
plt.show()


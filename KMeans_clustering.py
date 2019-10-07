# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:45:32 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=150, n_features=2, centers=3,cluster_std=0.5,
                  shuffle=True,random_state=0)
plt.scatter(X[:,0], X[:,1],c='white',marker='o',edgecolor='black',s=50)
plt.grid()
plt.show()

# init means the no. of initialization it has to made no. of times.
km=KMeans(n_init=100,init='random',n_clusters=3,max_iter=300,tol=1e-04,random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],s=50, c='lightgreen',
            marker='s', edgecolor='black',label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange',
            marker='o', edgecolor='black',label='cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue',
            marker='v', edgecolor='black', label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1],
            s=250, marker='*',c='red', edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

# kmeans++ is actually by default in the kmeans class box which we defined as the random in init class
# The distortion or SSE of the Clusters is as follow
print(km.cluster_centers_) # Gives cooridinate values
print(km.inertia_) # 72.47

# Compute a distortion vs no. of clusters graph for determining the best k clusters by elbow point
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,init='k-means++', n_init=10,max_iter=300,random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# Silhouette scores calculate the differnce b/w the cohesion and the saparation of a point divided by maximum of these two.
# Cohesion is the average distance of all the points in a cluster from a randomly chosen point
# Separation is the average distance of all the points in the very closest cluster from that same point
from sklearn.metrics import silhouette_samples
from matplotlib import cm

km = KMeans(n_clusters=3,init='k-means++',n_init=10,
            max_iter=300,tol=1e-04,random_state=0)
y_km = km.fit_predict(X)
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals=silhouette_samples(X,y_km,metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette=silhouette_vals[y_km==c]
    c_silhouette.sort()
    y_ax_upper+=len(c_silhouette)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),c_silhouette,color=color,height=1)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()
# As no value is close to 0  so this is a very good clustering
    
# Try a bad clustering by n_clusters=2 to see impact on silhouette plot
km=KMeans(n_init=100,init='random',n_clusters=2,max_iter=300,tol=1e-04,random_state=0)
y_km = km.fit_predict(X)    
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],s=50, c='lightgreen',
            marker='s', edgecolor='black',label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange',
            marker='o', edgecolor='black',label='cluster 2')
plt.scatter(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1],
            s=250, marker='*',c='red', edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
    
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals=silhouette_samples(X,y_km,metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette=silhouette_vals[y_km==c]
    c_silhouette.sort()
    y_ax_upper+=len(c_silhouette)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),c_silhouette,color=color,height=1)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()   
# Here the different or unsual length and width plus the silhouette values clearly indicates the bad clusteirng


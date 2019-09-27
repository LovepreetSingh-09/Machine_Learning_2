# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:37:43 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist,squareform
from scipy.linalg import eigh
from scipy import exp


def rbf_kernel_pca(X,n_components,gamma):
   # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    pdis=pdist(X,'sqeuclidean')
    print(pdis.shape)
    # Convert pairwise distances into a square matrix.
    sq=squareform(pdis)
    print(sq.shape)
    # Compute the symmetric kernel matrix.
    K=exp(-gamma*sq)
    print(K.shape)
    # Center the kernel matrix.
    N=K.shape[0]
    ln=np.ones((N,N))/N
    K=K-ln.dot(K)+ln.dot(K)-K.dot(ln).dot(ln)
    print(K.shape)
    eigen_val,eigen_vec=eigh(K)
    print(eigen_vec.shape)
    eigen_val,eigen_vec=eigen_val[::-1],eigen_vec[:,::-1]
    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigen_vec[:, i] for i in range(n_components)))
    return X_pc

# 2 half-moon dataset
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1],color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],color='blue', marker='o', alpha=0.5)
plt.show()
    
from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
# PCA did not separate data here instead it just make those 2 moons again with revertwd shape
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

# Now look how our very own kernel_pca perform on this data
print(X.shape)
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()
# Here our data was well linearly separated and we can use any linear classifier


# separating concentric circles:-
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000,random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0, 0], X[y==0, 1],color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],color='blue', marker='o', alpha=0.5)
plt.xlabel('')
plt.ylabel('')
plt.show()

# Now again in this data also standard pca didn't do anythhing to separate data
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

# Now look how our very own kernel_pca perform on this data
print(X.shape)
X_kpca = rbf_kernel_pca(X, gamma=7, n_components=2)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02,color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02,color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()
# Here our data is classified reasonably better we can use any linear classifier


# For Projecting new data points:-
def rbf_kernel_pca(X,n_components,gamma):
   # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    pdis=pdist(X,'sqeuclidean')
    print(pdis.shape)
    # Convert pairwise distances into a square matrix.
    sq=squareform(pdis)
    print(sq.shape)
    # Compute the symmetric kernel matrix.
    K=exp(-gamma*sq)
    print(K.shape)
    # Center the kernel matrix.
    N=K.shape[0]
    ln=np.ones((N,N))/N
    K=K-ln.dot(K)+ln.dot(K)-K.dot(ln).dot(ln)
    print(K.shape)
    eigen_val,eigen_vec=eigh(K)
    print(eigen_vec.shape)
    eigen_val,eigen_vec=eigen_val[::-1],eigen_vec[:,::-1]
    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigen_vec[:, i] for i in range(n_components)))
    # Collect the corresponding eigenvalues
    lambdas = [eigen_val[i] for i in range(n_components)]
    return alphas, lambdas

X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
x_new = X[25]
print(x_new)
x_proj = alphas[25] # original projection
print(x_proj)
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

x_reproj = project_x(x_new, X,gamma=15, alphas=alphas, lambdas=lambdas)
print(x_reproj)

plt.scatter(alphas[y==0, 0], np.zeros((50)),color='red', marker='^',alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)),color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',label='remapped point X[25]',marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()

# Kernel_PCA in scikit-learn
from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2,kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

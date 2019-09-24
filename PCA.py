# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:10:50 2019

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
from sklearn.linear_model import LogisticRegression

df_wine = pd.read_csv('https://archive.ics.uci.edu/''ml/machine-learning-databases/''wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
print(df_wine.columns)
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.3,random_state=0,stratify=y)
print(X)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# PCA is an unsupervised method which has nothing to do with class labels unlike Feature Selection methods(SBS)
# PCA is used for dimensionality reduction where variance is used as the information.
# The k components with the most variance has been selected to make the k dimensions.
# Firstly, the covariance matrix is made from the input standardize data which is not having a unit becoz of divison with Std and all the features comes onto same scale with mean is 0.
print(X_train_std.shape) # (124, 13)
cov=np.cov(X_train_std.T) 
print(cov.shape) # (13,13)
# Here 13 X 13 cov matrix represent the variance where diagonal is having variance of each feature.
# Other cell represent the covariance b/w 2 features and if it is -ve ,it means there are oppositely corelated and +ve for direct corelated.
# This cov matrix will be used to find the eigen value and eigen vector .
# Eigen vectors of cov matrix represent the principal component(direction of maximum variation)
eigen_val,eigen_vect=np.linalg.eig(cov)
print(eigen_val)
print(eigen_vect[0])

# Eigen value represent the magnitude of eigen vectors  so we need to sort them
eigen_pair=[(np.abs(eigen_val[i]),eigen_vect[:,i]) for i in range(len(eigen_val))]
eigen_pair.sort(reverse=True) 
print(eigen_pair[12])

# Vaiance explained Ratio :
# It is the ratio of an eigen value divided by the total sum of eigen values.
tot=np.sum(eigen_val)
print(tot)
ver=[i/tot for i in sorted(eigen_val,reverse=True)]
print(ver)
cum_sum=np.cumsum(ver)
plt.bar(range(1,14),ver,align='center',label='individual explained variance')
plt.step(range(1,14),cum_sum,label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()
# As you can see from graph the first 2 components arre responsible for the 60 % of the information (variance) so lets choose only the 2 eigen_vlaues' corresponding vectors to compress the data into 2 features

# Now create a W matrix consist of k no. of top vectors (here k=2).
W=np.hstack([eigen_pair[0][1][:,np.newaxis],eigen_pair[1][1][:,np.newaxis]])
print(W.shape) # (13, 2)

# Now by dot product of W with the original input dataset we will get our new dataset with 2 features
X_pca=np.dot(X_train_std,W)
print(X_pca.shape) #(124, 2)
# W can also be used for getting back original data by having dot product of transpose of W with X_pca but that data will have some noise

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_pca[y_train==l, 0],X_pca[y_train==l, 1], c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
# As we can see from graph data is more spread in x-axis (first principal component) and now we can use a linear classifier to classify this pca data

# PCA method is also implemented in Scikit-Learn
from sklearn.decomposition import PCA
def plot_decision_regions(X,y,classifier,res=0.02,test_idx=None):
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    marker=['*','s','*','^']
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,res),np.arange(x2_min,x2_max,res))
    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z=z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.5,cmap=cmap)
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],marker=marker[idx],cmap=cmap,c=colors[idx],alpha=0.8,label=cl)
    if test_idx:
        X_test,y_test=X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='',s=120,edgecolors='black',alpha=1.0,marker='o',label='test')

pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
X_combined=np.vstack([X_train_pca,X_test_pca])
y_combined=np.hstack([y_train,y_test])
lr.fit(X_train_pca, y_train)
print('Accuracy :',accuracy_score(lr.predict(X_test_pca),y_test)) # 0.9260
plot_decision_regions(X_combined, y_combined, classifier=lr,test_idx=range(100,124))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

print(pca.explained_variance_ratio_) # This is of just 2 components that we gave to it
# Now lets see evr for all features
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)

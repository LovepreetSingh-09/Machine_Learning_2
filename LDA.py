# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 22:29:05 2019

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

# LDA is Linear Discriminant Analysis.
# It is a supervised technique for dimensionality reduction for the class labels.
# It tries to increase the difference b/w the means of 2 or more class and reducing the variance of each class
# The assumptions in this is that data is is normally distributed, classes have identical covariance matrix and features are independent of each other
# Even after violating these assumption slightly(mostly), we still get reduce dimensions weell

# After standardizing the data, compute the mean of each original dimension or feature of each class.
mean_vec=[]
print(np.unique(y_train)) # [1 2 3]
for label in range(1,4):
    mean_vec.append(np.mean(X_train_std[y_train==label],axis=0))

print(mean_vec)
d=X_train_std.shape[1]
    
# Now we will construct individual scatter matrix to construct within-class scatter matrix
sw=np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vec):
    si=np.zeros((d,d))
    for row in X_train_std[y_train==label]:
        row,mv=row.reshape(d,1),mv.reshape(d,1)
        si+=(row-mv).dot((row-mv).T)
    sw+=si

print('Within-class Scatter Matrix : ',sw.shape)
print(sw[0])

# We made an assumption that the class labels are uniformly distributed but actual are-
print(np.bincount(y_train))      # [41 50 33]
# So, now we should scale the individual scatter matrix.
# When we divide them by the no. of class samples , we found that the formula for within-class matrix becomes the formula of covariance matrix.
# So this is the 2nd way to compute within-class scatter matrix straightly by covariance.
# Here covariance matrix is the si for each class which then add-up to make sw
sw=np.zeros((d,d))
for label in range(1,4):
    si=np.cov(X_train_std[y_train==label].T)
    sw+=si

# Now we have sacled sw
print(sw[0])

# Now, we have to compute the b/w-class scatter matrix by the formula where difference of mean of each class and the overall mean is taken into the account.
sb=np.zeros((d,d))
m=np.mean(X_train_std,axis=0)
m=m.reshape(d,1)
for labels,mv in zip(range(1,4),mean_vec):
    n=X_train_std[y_train==label].shape[0]
    mv=mv.reshape(d,1)
    sb+=n*(mv-m).dot((m-mv).T)

print(sb[0])
    
# Now, the remaning part is much like PCA.
# The differnce here is that we have to find the eigen_val,eigen_vec for the dot product of inverse of sw and sb
eigen_val,eigen_vect=np.linalg.eig(np.linalg.inv(sw).dot(sb))
print(eigen_val[5].real)

eigen_pair=[(np.abs(eigen_val[i]),eigen_vect[:,i]) for i in range(len(eigen_val))]
eigen_pair=sorted(eigen_pair,key=lambda k : k[0],reverse=True)
print(eigen_pair[:][0])
for eigen_vals in eigen_pair:
    print(eigen_vals[0])

# As you can see the top two val are very high while others are almost zero (actually zero) because computational methods
# Because in LDA the no. of linear discriminants are c-1 (c=class labels)

# Now construct W matrix of top 2 eigen vec
w = np.hstack((eigen_pair[0][1][:, np.newaxis].real,eigen_pair[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1] * (-1),c=c, label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()
# Here the data is well classified on 2D plane.

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

X_test_lda = lda.transform(X_test_std)
print(lr.score(X_test_std,y_train.reshape(1,-1)))
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
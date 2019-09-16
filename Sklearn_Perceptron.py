# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:32:11 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from matplotlib.colors import ListedColormap
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris=load_iris()
print(iris.keys())
print(np.unique(iris.target))
X_train,X_test,y_train,y_test=train_test_split(iris.data[:,[2,3]],iris.target,test_size=0.3,random_state=1,stratify=iris.target)
print(np.bincount(y_train))
print(np.bincount(y_test))
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
ppn=Perceptron(random_state=1,eta0=0.1,n_iter=1000)
ppn.fit(X_train_std,y_train)
y_pred=ppn.predict(X_test_std)
print('Misclassified Samples:',np.sum(y_pred!=y_test))
print('Accuracy: %.2f'%accuracy_score(y_test,y_pred))

def decision_plot_regions(X,y,classifier,res=0.02,test_idx=None):
    colors=['green','blue','red','yellow']
    marker=['*','x','s','^']
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,res),np.arange(x2_min,x2_max,res))
    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z=z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.5)
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],marker=marker[idx],cmap=cmap,c=colors[idx],alpha=0.8)
    if test_idx:
        X_test,y_test=X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='',s=120,edgecolors='black',alpha=1.0,marker='o')

X_combined=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
decision_plot_regions(X_combined,y_combined,classifier=ppn,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()




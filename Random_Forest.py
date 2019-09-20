# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:20:30 2019

@author: user
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

iris=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris.data[:,[2,3]],iris.target,test_size=0.3,random_state=1,stratify=iris.target)
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_combined=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

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


tree=RandomForestClassifier(n_estimators=25,n_jobs=1,random_state=1,criterion='entropy')
tree.fit(X_train,y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
y_pred=tree.predict(X_test)
print(tree.score(X_test,y_test)) # 97.77 % using both gini andentropy
plot_decision_regions(X_combined,y_combined,classifier=tree,test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend()
plt.show()


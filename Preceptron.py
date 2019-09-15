# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:33:42 2019

@author: user
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from matplotlib.colors import ListedColormap


class Preceptron:
    def __init__(self,n_iter=100,eta=0.05):
        self.n_iter=n_iter
        self.eta=eta
        
    def fit(self,X,y):
        self.Errors=[]
        self.w_=np.random.RandomState(1).normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        for i in range(self.n_iter):
            error=0
            for xi,yi in zip(X,y): 
                update=self.eta*(yi-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                error+=int(update!=0.0)
            self.Errors.append(error)
    def input_(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    
    def predict(self,X):
        return np.where(self.input_(X)>=0.0,1,-1)
        
data=pd.read_csv('iris.data')
display(data.head())
display(data.columns)
y=data.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',1,-1)
X=data.iloc[0:100,[0,2]].values
print(X[:,1])
print(y[0])
#mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.scatter(X[:50,0],X[:50,1],c='b',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],c='g',marker='*',label='versicular')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.legend()
plt.show()

ppn=Preceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
print(ppn.predict(np.array([2.5,7.0]))) # -1
plt.plot(range(len(ppn.Errors)),ppn.Errors)
plt.show()

#v1=np.array([1,2,3])
#v2=0.5*v1
#print(np.arccos(v1.dot(v2)/np.linalg.norm(v1)*np.linalg.norm(v2)))
def plot_decision_regions(X, y, classifier, res=0.02):
    x1_min,x1_max=min(X[:,0])-1,max(X[:,0])+1
    x2_min,x2_max=min(X[:,1])-1,max(X[:,1])+1
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,res),np.arange(x2_min,x2_max,res))
    print(xx1.shape,xx2.shape) # (350, 235) (350, 235)
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    print(Z.shape) # (82250,)
    Z = Z.reshape(xx1.shape)
    print(Z.shape)  # (350, 235)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
print(X[y == -1, 0]) # [7.  6.4 6.9 5.5 6.5 5.7 6.3 4.9 6.6 5.2 5.  5.9 6.  6.1 5.6 6.7 5.6 5.8
 # 6.2 5.6 5.9 6.1 6.3 6.1 6.4 6.6 6.8 6.7 6.  5.7 5.5 5.5 5.8 6.  5.4 6.
 # 6.7 6.3 5.6 5.5 5.5 6.1 5.8 5.  5.6 5.7 5.7 6.2 5.1 5.7 6.3]

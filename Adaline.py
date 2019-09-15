# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:15:55 2019

@author: user
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from matplotlib.colors import ListedColormap

#  This is an example of Batch Gradient Descent
class adaline(object):
    def __init__(self,eta=0.01,n_iter=10,randomstate=1):
        self.eta=eta
        self.n_iter=n_iter
        self.RandomState=randomstate
    def fit(self,X,y):
        self.w_=np.random.RandomState(self.RandomState).normal(loc=0.0,scale=0.1,size=1+X.shape[1])
        self.errors=[]
        for i in range(self.n_iter):
            inp=self.input_(X)
            pred=self.activation(inp)
            update=self.eta*(y-pred)
            self.w_[0]+=update.sum()
            self.w_[1:]+=X.T.dot(update)
            cost=(((y-pred)**2).sum())/2
            self.errors.append(cost)
    def input_(self,X):
        return X.dot(self.w_[1:])+self.w_[0]
    def activation(self,X):
        return X
    def predict(self,X):
        return np.where(self.activation(self.input_(X))>=0.0,1,-1)

data=pd.read_csv('iris.data')
display(data.head())
display(data.columns)
y=data.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',1,-1)
X=data.iloc[0:100,[0,2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1=adaline(eta=0.01,n_iter=10)
ada1.fit(X,y)
print(ada1.errors)
ax[0].plot(range(1, len(ada1.errors) + 1),np.log10(ada1.errors), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = adaline(n_iter=10, eta=0.0001)
ada2.fit(X, y)
print(ada2.errors)
ax[1].plot(range(1, len(ada2.errors) + 1),ada2.errors, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
            
def plot_decision_regions(X,y,classifier,res=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,res),np.arange(x2_min,x2_max,res))
    print(xx1.shape)
    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z=z.reshape(xx1.shape)
    print(z.shape)
    plt.contourf(xx1,xx2,z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],marker=markers[idx],c=colors[idx],label=cl)

Xstd=np.copy(X)      
# Standardization:- 
Xstd[:,0]=(Xstd[:,0]-np.mean(Xstd[:,0]))/np.std(Xstd[:,0])  
Xstd[:,1]=(Xstd[:,1]-np.mean(Xstd[:,1]))/np.std(Xstd[:,1])   

ada = adaline(n_iter=15, eta=0.01)
ada.fit(Xstd, y)
plot_decision_regions(Xstd, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

        
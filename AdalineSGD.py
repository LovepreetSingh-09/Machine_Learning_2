# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:16:20 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from matplotlib.colors import ListedColormap

# Stochastic Gradient Descent
class AdalineSGD(object):
    def __init__(self,eta=0.01,n_iter=10,shuffle=True):
        self.eta=eta
        self.n_iter=n_iter
        self.shuffle=shuffle
        self.weights_initialized=False
        self.randomstate=1
    def fit(self,X,y):
        self.weights(X.shape[1])
        self.cost=[]
        for i in range(self.n_iter):
            if self.shuffle:
                X,y=self.shuffle_(X,y)
            c=[]
            for xi,yi in zip(X,y):
                c.append(self.update_weights(xi,yi))
            avgc=sum(c)/len(y)
            self.cost.append(avgc)
        return self
                
    def update_weights(self,xi,yi):
        output=self.activation(self.input_(xi))
        costf=(yi-output)
        self.w_[0]+=costf*self.eta
        self.w_[1:]+=xi.dot(costf)*self.eta
        costf=0.5*costf**2
        return costf
    
    def partial_fit(self,X,y):
        if not self.weights_initialized:
            self.weights(X.shape[1])
        if y.ravel().shape[0]>1:
            for xi,yi in zip(X,y):
                self.update_weights(xi,yi)
        else:
            self.update_weights(X,y)
        return self
    
    def weights(self,m):
        self.w_=np.random.RandomState(self.randomstate).normal(loc=0.0,scale=0.01,size=1+m)
   
    def shuffle_(self,X,y):
        r=np.random.RandomState(self.randomstate).permutation(len(y))
        return X[r],y[r]
    
    def activation(self,inp):
        return inp
    
    def input_(self,X):
        return X.dot(self.w_[1:])+self.w_[0]
    
    def predict(self,X):
        return np.where(self.activation(self.input_(X))>=0.0,1,-1)
    
data=pd.read_csv('iris.data')
display(data.head())
display(data.columns)
y=data.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',1,-1)
X=data.iloc[0:100,[0,2]].values
Xstd=np.copy(X)      
# Standardization:- 
Xstd[:,0]=(Xstd[:,0]-np.mean(Xstd[:,0]))/np.std(Xstd[:,0])  
Xstd[:,1]=(Xstd[:,1]-np.mean(Xstd[:,1]))/np.std(Xstd[:,1])   
print(Xstd.shape)

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

ada = AdalineSGD(n_iter=15, eta=0.01)
ada.fit(Xstd, y)
print(ada.cost)
plot_decision_regions(Xstd, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada.cost) + 1), ada.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
    
    






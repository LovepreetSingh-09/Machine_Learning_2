# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:58:53 2019

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
from sklearn.linear_model import LogisticRegression

def sigmoid(x):
    return 1/(1+np.exp(-x))
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.show()

def cost1(x):
    return -np.log(sigmoid(x))
def cost0(x):
    return -np.log(1-sigmoid(x))
a=np.arange(-10,10,0.1)
z=sigmoid(a)
plt.plot(z,cost1(a),c='blue',label='y=1')
plt.plot(z,cost0(a),c='red',linestyle='--',label='y=0')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.show()

class Logistic_Regression(object):
    def __init__(self,n_iter=20,eta=0.01,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
    def fit(self,X,y):
        regn=np.random.RandomState(self.random_state)
        self.w_=regn.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        error=0
        self.cost_=[]
        for i in range(self.n_iter):
            inp=self.input_(X)
            output=self.activation(inp)
            error=y-output
            self.w_[1:]+=self.eta*X.T.dot(error)
            self.w_[0]+=self.eta*(error.sum())
            cost=(-y.dot(np.log(output)))-((1-y).dot(np.log(1-output)))
            self.cost_.append(cost)
        return self
    def input_(self,X):
        return X.dot(self.w_[1:])+self.w_[0]
    def activation(self,x):
        return 1/(1+np.exp(-np.clip(x,-250,250))) # Here whatever the value of z is obtained but for the exponential the value between -250 to 250 will be given
    def predict(self,X):
        return np.where(self.activation(self.input_(X))>=0.5,1,0)

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


iris=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris.data[:,[2,3]],iris.target,test_size=0.3,random_state=1,stratify=iris.target)
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_train_sub=X_train_std[(y_train==0)|(y_train==1)]
y_train_sub=y_train[(y_train==0)|(y_train==1)]
print(X_train_sub.shape)
print(np.bincount(y_train_sub)) # [35 35]
logreg=Logistic_Regression(eta=0.05,n_iter=1000,random_state=1)
logreg.fit(X_train_sub,y_train_sub)
decision_plot_regions(X_train_sub,y_train_sub,classifier=logreg)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

logreg=LogisticRegression(C=100,random_state=1)
logreg.fit(X_train_std,y_train)
X_combined=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
decision_plot_regions(X_combined,y_combined,classifier=logreg,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

print(logreg.predict_proba(X_train_std[:3,:]))
print(logreg.predict_proba(X_train_std[:3,:]).argmax(axis=1))
print(logreg.predict(X_train_std[:3,:]))
# Always give 2D array while using predict of sklearn
print(logreg.predict(X_train_std[2,:].reshape(1,-1)))

# Regularization:-
# C is inverse of regularization parameter lambda 
# For regularization , 0.5*lambda*|w|**2 has been added to avoid overfitting
# Hence, the weight update is lesser and thus it reduces the chance of overfitting
weights,params=[],[]
for c in np.arange(-5,5):
    logreg=LogisticRegression(C=10.**c,random_state=1)
    logreg.fit(X_train_std,y_train)
    b=logreg.coef_.shape
    weights.append(logreg.coef_[1])
    params.append(10.**c)
    
weights=np.array(weights)
print(b)  # coef_ is actually the weights per class per feature. Here, its shape is (3,2) 3 classes 2 features
print(weights)
plt.plot(params,weights[:,0],label='Petal Length')
plt.plot(params,weights[:,1],linestyle='--',label='Petal Width')
plt.ylabel('Weights coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend()
plt.show()


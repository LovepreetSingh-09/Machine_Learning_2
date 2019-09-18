# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:31:52 2019

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
from sklearn.svm import SVC

iris=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris.data[:,[2,3]],iris.target,test_size=0.3,random_state=1,stratify=iris.target)
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_combined=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

def decision_plot_regions(X,y,classifier,res=0.02,test_idx=None):
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

# C is the penalty parameter which penalize the model after a misclassification.
# More the C , more will be the penalty after every misclassification and tighter will be the margin
# There is also a slack variable which is used in objective function with C.
svm=SVC(C=1,kernel='linear',random_state=1)
svm.fit(X_train_std,y_train)
decision_plot_regions(X_combined,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# Kernel Trick:-
# For the non-linear classification problem kernel tricks can be used in SVM 
# Default Kernel is "RBF" while there are no. of kernels like rbf,polynomial,sgd,linear etc.
# Lets create a non-linear classified data with XOR where only and only one feature per sample needs to be True
np.random.seed(1)
X=np.random.randn(200,2)
y=np.logical_xor(X[:,0]>0,X[:,1]>0)
y=np.where(y,1,-1)
plt.scatter(X[y==1,0],X[y==1,1],marker='s',c='red')
plt.scatter(X[y==-1,0],X[y==-1,1],marker='*',c='blue')
plt.xlabel('')
plt.ylabel('')
plt.title('XOR')
plt.show()

# The kernel tricks tries to classify data by extending the features
# Here rbf adds a new feature which is made from first 2 features by Gaussian kernel formula e**(-gamma*(x-center))
svm=SVC(kernel='rbf',gamma=0.1,C=10,random_state=1)
svm.fit(X,y)
decision_plot_regions(X,y,classifier=svm)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

svm=SVC(C=1,kernel='rbf',gamma=0.2,random_state=1)
svm.fit(X_train_std,y_train)
decision_plot_regions(X_combined,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# Here more the gamma, more will be the reach and influence of the of training data and makes a tighter boundaries
# Then there will definately be very high error on the test data
svm=SVC(C=1,kernel='rbf',gamma=100,random_state=1)
svm.fit(X_train_std,y_train)
decision_plot_regions(X_combined,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()




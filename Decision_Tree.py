# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 22:38:56 2019

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
import graphviz

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

# In Decision tree there are various algorithms like CART , id3, id4.5, CHAID etc.
# In scikit-learn CART is used. 
# It is based on percentage of impurities in the node where 100 % impurity means the classes are equally present while 0% means only 1 class is there.
# For impurities there is gini index, entropy etc.
# Maximum range of gini is 0-0.5 while entropy has range 0-1

def gini(p):
    return 1-(p**2 + (1-p)**2)
def entropy(p):
    return -p*np.log2(p)-(1-p)*np.log2(1-p)
def maxe(p):
    return 1-np.max([p,1-p])
x=np.arange(0.0,1,0.01)
ent=[entropy(g) if g!=0 else None for g in x] 
print(len(ent))
gin=gini(x)
print(gin.shape)
ma=[maxe(i) for i in x] 
print(len(ma))
sc_ent=[p*0.5 if p else None for p in ent]
fig=plt.figure()
ax=plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gin, ma],['Entropy', 'Entropy (scaled)','Gini Impurity','Misclassification Error'],['-', '-', '--', '-.'],['black', 'lightgray','red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab,linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.15,1.15),shadow=True,ncol=5)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

# Here 1 thing to notice that we can't use the standardized or normalized features because for making a decision there should be some dissimilarity b/w the features of a dataset
tree=DecisionTreeClassifier(criterion='gini',random_state=1,max_depth=4)
tree.fit(X_train,y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
y_pred=tree.predict(X_test)
print(tree.score(X_test,y_test)) # 97.77 % using gini and 93.33 % using entropy
plot_decision_regions(X_combined,y_combined,classifier=tree,test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend()
plt.show()

dot_file=export_graphviz(tree,filled=True,rounded=True,class_names=['Setosa','Versicolor','Virginica'],feature_names=['Petal Width','Petal Length'],out_file=None)
graph=graph_from_dot_data(dot_file)
graph.write_png('tree.png')



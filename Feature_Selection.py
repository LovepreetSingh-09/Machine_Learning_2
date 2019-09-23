# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:30:33 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from sklearn.base import clone
from itertools import combinations
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df_wine = pd.read_csv('https://archive.ics.uci.edu/''ml/machine-learning-databases/''wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
print(df_wine.columns)
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())
# iloc[ ] is used for the slicing function.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.3,random_state=0,stratify=y)
print(X)

# L1 regularization usually yields sparse feature vectors and most feature weights will be zero.
# Sparsity can be useful in practice if we have a high-dimensional dataset with many features that are irrelevant, especially cases where we have more irrelevant dimensions than samples. In this sense, L1 regularization can be understood as a technique for feature selection.
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# SBS means sequential backward selection to select the most important features by neglecting the least important features.

class SBS(object):
    def __init__(self,k_feat,estimator,random_state=1,test_size=0.25,scoring=accuracy_score):
        self.k_feat=k_feat
        self.random_state=random_state
        # clone does the deep copy of estimator with same parameters
        self.estimator=clone(estimator)
        self.scoring=accuracy_score
        self.test_size=test_size
        
    def fit(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)
        dim=X.shape[1]
        self.scores=[]
        self.indices=tuple(range(dim))
        self.subset=[self.indices]
        s=self.calc_score(X_train,X_test,y_train,y_test,indices=self.indices)
        self.scores.append(s)
        while dim>self.k_feat:
            score=[]
            subset=[]
            # combinations always take tuple as its first argument
            for p in combinations(self.indices,r=dim-1):
                sc=self.calc_score(X_train,X_test,y_train,y_test,indices=p)
                score.append(sc)
                subset.append(p)
            best=np.argmax(score)
            self.indices=subset[best]
            self.scores.append(score[best])
            self.subset.append(subset[best])
            dim=dim-1
        self.k_score=self.scores[-1]
        return self
    
    def transform(self,X):
        return X[:,self.indices]
                
    def calc_score(self,X_train,X_test,y_train,y_test,indices):
        m=self.estimator.fit(X_train[:,indices],y_train)
        sco=m.predict(X_test[:,indices])
        score=self.scoring(sco,y_test)
        return score
    
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(estimator=knn, k_feat=1)
sbs.fit(X_train_std, y_train)

# plotting the graph between accuracy and the no. of features
sv=[len(k) for k in sbs.subset]
plt.plot(sv,sbs.scores,marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k3=list(sbs.subset[10])
# shows the 3 most important features
print(df_wine.columns[1:][k3])

# Checking accuracy with all the fatures
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

# Now check the accuracy with 3 most important features
knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:',knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:',knn.score(X_test_std[:, k3], y_test))
# As you can notice that the accuracy is slightly lesser but we used just 3 fetures out of 13
# So, in this way we can reduce the computational cost and time with the expense of slight accuracy

# Feature importance with random Forest
# Scikit Learn Provides a grat feature in Random Forest to tell the importance of each feature
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train, y_train)
importance=forest.feature_importances_
imp=np.argsort(importance)[::-1]
print(imp)

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[imp[f]],importance[imp[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importance[imp],align='center')
plt.xticks(range(X_train_std.shape[1]),feat_labels[imp],rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


# Sklearn also provide a special function SelectFromModel which gives the feature which are above a user-specified threshold.
# We also need to provide a model
from sklearn.feature_selection import SelectFromModel
md=SelectFromModel(forest,threshold=0.1,prefit=True)
X_selected=md.transform(X_train)
print('Number of samples that meet this criterion:',X_selected.shape[0])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[imp[f]],importance[imp[f]]))

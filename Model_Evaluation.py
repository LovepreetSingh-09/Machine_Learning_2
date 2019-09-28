# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:22:56 2019

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
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases''/breast-cancer-wisconsin/wdbc.data',header=None)
print(df.head());df.columns
X=df.iloc[:,2:].values;X.shape
# 1st column is of id so we would neglect that
y=df.iloc[:,1].values; y.shape,np.unique(y)

le=LabelEncoder()
y=le.fit_transform(y)
print(le.classes_)
print(le.transform(['M','B']))

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=1)

# We can use all preprocessing functions in the pipeline but at last there should be an estimator.
pipe=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
pipe.fit(X_train,y_train)
print(pipe.predict(X_test))
print(pipe.score(X_test,y_test)) # 0.956

# Stratified Kfold Validation
# Stratified makes folds based on the equal distribution of classes in each fold
from sklearn.model_selection import StratifiedKFold
kfold=StratifiedKFold(n_splits=10,random_state=1).split(X_train,y_train)

score=[]
for k,(train,test) in enumerate(kfold):
    pipe.fit(X_train[train],y_train[train])
    scores=pipe.score(X_train[test],y_train[test])
    score.append(scores)
    print('Fold: %2d, Class distribution: %s, Acc: %.3f' % (k+1,np.bincount(y_train[train]), scores))

print(score)
print('CV Accuracy : ',np.mean(score),' CV Std : ',np.std(score)) # 0.949 , 0.013

# Instead of writing this much code for Validation we can use cross_val_score of scikit-learn
from sklearn.model_selection import cross_val_score
scores=cross_val_score(estimator=pipe,X=X_train,y=y_train,cv=10,n_jobs=1)
print(np.mean(scores)) # 0.949
# We got the same score by this function

# Learning Curve:-
# To visualize the best bias-variance trade based on the training data size with training and validation scores.
# in case of underfitting and overfitting data size makes the most effect
from sklearn.model_selection import learning_curve
# Here the training data will used to make model with the pipeline and Validation with the size increase by 10% every time
train_sizes,train_scores,test_scores=learning_curve(estimator=pipe,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1,10),cv=10,n_jobs=1)
print(train_sizes)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean,color='blue', marker='o',markersize=5, label='training accuracy')
plt.fill_between(train_sizes,train_mean + train_std,train_mean - train_std,alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()
# Lesser the difference betweenthe training and testing curve better the model will generalize to unseen data

# Validation Curve:-
# It is very much similar to lerning curve with a little difference
# Instead of training data size, the parameters are checked for better generalization
from sklearn.model_selection import validation_curve
param_range=[0.001,0.01,0.1,1,10,100]
train_scores,test_scores=validation_curve(estimator=pipe,X=X_train,y=y_train,param_name='logisticregression__C',param_range=param_range,cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean,color='blue', marker='o',markersize=5, label='training accuracy')
plt.fill_between(param_range,train_mean + train_std,train_mean - train_std,alpha=0.15, color='blue')
plt.plot(param_range, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
plt.fill_between(param_range,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.xlabel('C (Inverse Regularization Parameter)')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.show()


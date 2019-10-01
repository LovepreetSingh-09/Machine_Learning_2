# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:42:44 2019

@author: user
"""
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/wine/wine.data',header=None)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash','Magnesium',
'Total phenols','Flavanoids', 'Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']

 # drop  class label 1 
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol','OD280/OD315 of diluted wines']].values

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2,random_state=1,stratify=y)

from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=None)
# Once the individual classifiers are fit to the bootstrap samples, the predictions are combined using majority voting.
# Ensemble of 500 tree and always use float number 1.0 in parameters
bag=BaggingClassifier(base_estimator=tree,n_estimators=500,max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=1,random_state=1)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))  # 1.000/0.833
# As it is cleared that our tree model has overfitted the data becoz of so much difference of train and test accuracies

bag=bag.fit(X_train,y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f'% (bag_train, bag_test)) # 1.000/0.917
# We have got much better accuracy on the test data

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,sharex='col',sharey='row',figsize=(8, 3))
for idx, clf, tt in zip([0, 1],[tree, bag], ['Decision tree', 'Bagging']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],X_train[y_train==0, 1],c='blue',marker='^',s=50)
    axarr[idx].scatter(X_train[y_train==1, 0],X_train[y_train==1, 1],c='green',marker='o',s=50)
    axarr[idx].set_title(tt)
plt.text(10.2, -1.2, s='OD280/OD315 of diluted wines',ha='center', va='center', fontsize=12)
plt.show()
# Although Bagging Classifier i more complex but it produces better results.


# Adaptive boosting :-
# After doing adaptive boosting with weak learners , weak learrners are combined for majority voting for prediction
# The weights of misclassified samples in the previous weak learners being increased
from sklearn.ensemble import AdaBoostClassifier

tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=1)
ada=AdaBoostClassifier(base_estimator=tree,n_estimators=500,learning_rate=0.1,random_state=1)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test)) # 0.916/0.875

ada.fit(X_train,y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f'% (ada_train, ada_test)) # 1.000/0.917
# We got better accuracy in AdaBoost but there is still a chance that we have overfitted the model

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,sharex='col',sharey='row',figsize=(8, 3))
for idx, clf, tt in zip([0, 1],[tree, ada], ['Decision Tree', 'AdaBoost']):    
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],X_train[y_train==0, 1],c='blue',marker='^',s=50)
    axarr[idx].scatter(X_train[y_train==1, 0],X_train[y_train==1, 1],c='green',marker='o',s=50)
    axarr[idx].set_title(tt)
plt.text(10.2, -0.5,s='OD280/OD315 of diluted wines',ha='center',va='center',fontsize=12)
plt.show()


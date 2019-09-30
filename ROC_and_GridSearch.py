# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:21:14 2019

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
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

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

pipe=make_pipeline(StandardScaler(),SVC())
param_range=[0.001,0.01,0.1,1,10,100]
param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__kernel':['rbf'],'svc__gamma':param_range}]
gr=GridSearchCV(estimator=pipe,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=1)

gr.fit(X_train,y_train)
print(gr.best_score_)  # 0.984
print(gr.best_params_) # {'svc__C': 100, 'svc__gamma': 0.001, 'svc__kernel': 'rbf'}
print(gr.best_estimator_,gr.best_index_) # 36

clf=gr.best_estimator_
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))  # 0.9736

# Nested Cross-validation by GridSearch
gs=GridSearchCV(pipe,param_grid=param_grid,cv=10,n_jobs=-1)
fv=cross_val_score(gs,X_train,y_train,cv=10)
print(np.mean(fv)) # 0.9780
print(np.std(fv))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

from sklearn.metrics import recall_score,precision_score,f1_score
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

from sklearn.metrics import make_scorer
scorer=make_scorer(f1_score,pos_label=0)
gs = GridSearchCV(estimator=pipe,param_grid=param_grid,scoring=scorer,cv=10)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

# ROC means reciever operating curve and AUC means area under curve
# ROC graph is FPR v/s Recall or TPR
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(penalty='l2',random_state=1,C=100.0))
# Picking up only 2 features
X_train2 = X_train[:, [4, 14]]
cv = list(StratifiedKFold(n_splits=3,random_state=1).split(X_train,y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])
    # roc_cuve returns FPR, TPR and Threshold
    fpr, tpr, thresholds = roc_curve(y_train[test],probas[:, 1],pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,tpr,label='ROC fold %d (area = %0.2f)'% (i+1, roc_auc))
plt.plot([0, 1],[0, 1],linestyle='--',color=(0.6, 0.6, 0.6),label='random guessing')
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],[0, 1, 1],linestyle=':',color='black',label='perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")
plt.show()

# Micro and Macro scorings an be used as per situation. Macro is default
pre_scorer = make_scorer(score_func=precision_score,pos_label=1,greater_is_better=True,average='micro')

# Lets create a 9:1 imbalanced dataset
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))
y_pred = np.zeros(y_imb.shape[0])
# We are getting 90% accuacy by always predicting 0 class
print(np.mean(y_pred == y_imb) * 100)

from sklearn.utils import resample
X_new,y_new=resample(X_imb[y_imb==1],y_imb[y_imb==1],replace=True,n_samples=X_imb[y_imb==0].shape[0],random_state=123)


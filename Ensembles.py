# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:37:26 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import comb
import math

def ensemble_error(n_clf,error):
    k_start=math.ceil(n_clf/2)
    prob=[comb(n_clf,k)*(error**k)*(1-error)**(n_clf-k) for k in range(k_start,n_clf+1)]
    return sum(prob)
print(ensemble_error(n_clf=11, error=0.25)) # 0.03432

# The graph of base/individual classifier error v/s ensemble error
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_clf=11, error=error) for error in error_range]
plt.plot(error_range, ens_errors,label='Ensemble error',linewidth=2)
plt.plot(error_range, error_range,linestyle='--', label='Base error',linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()


print(np.argmax(np.bincount([0, 0, 1],weights=[0.2, 0.2, 0.6]))) # 1
ex = np.array([[0.9, 0.1],[0.8, 0.2],[0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
print(p) # [0.58 0.42]
print(np.argmax(p))  # 0

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    
    def __init__(self, classifiers,vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        
    def fit(self, X, y):
        # Use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
            return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),axis=1)
        else: 
            # 'classlabel' vote
            # Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,weights=self.weights)),axis=1,arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
    
    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas,axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
        for name, step in six.iteritems(self.named_classifiers):
            for key, value in six.iteritems(step.get_params(deep=True)):
                out['%s__%s' % (name, key)] = value
        return out

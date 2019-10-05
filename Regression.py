# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:23:19 2019

@author: user
"""

import pyprind
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/''python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD',
                 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2)
plt.tight_layout()
plt.show()

# Pearson Corelation
cm=np.corrcoef(df[cols].values.T)
print(cm.shape) # 5 X 5
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.show()

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return self.net_input(X)

X = df[['RM']].values
y = df['MEDV'].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
# Std requires 2D data so after transform convert into 1D
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
print(np.min(y_std))
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

sns.reset_orig() # resets matplotlib style
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()


def lin_regplot(X,y,model):
    plt.scatter(X,y,c='steelblue',edgecolor='w',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()
    
# inverse transform to go back to original values
num_rooms_std = sc_x.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))

print('Slope: %.3f' % lr.w_[1]) #  0.695
print('Intercept: %.3f' % lr.w_[0]) # -0.000
    
# The advance algorithms like in sklearn works well on unstandardized data
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0]) # 9.102
print('Intercept: %.3f' % slr.intercept_) # -34.671
print(slr.predict([[6]]))

# Here the results are pretty much same even without standardizing the data
lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

# Finding weights by formula 
# We will get the same weights as LinearRegression
# adding a column vector of "ones"
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print('Slope: %.3f' % w[1]) # 9.102
print('Intercept: %.3f' % w[0]) # -34.671

# RANdom SAmple Consensus (RANSAC) algorithm :-
from sklearn.linear_model import RANSACRegressor
ransac=RANSACRegressor(LinearRegression(),max_trials=100,min_samples=50,residual_threshold=5.0,random_state=0,loss='absolute_loss')
ransac.fit(X,y)
# We get the array of data points in boolean whether those are inlier or outlier
inlier_mask=ransac.inlier_mask_
print(inlier_mask)
# Same array to make outlier in the previous True and inliers as False
outlier_mask=np.logical_not(inlier_mask)
print(outlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],c='steelblue', edgecolor='white',marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white',marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()

# Weights of ransac Regressor
# As we can see here the weights are quite different from the previous model and the more inliers are taken into account
print(ransac.estimator_.coef_) # [10.73450881]
print(ransac.estimator_.intercept_) # -44.08906428639814

# Evaluating the model
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# Residual Plot :-
# residuals (the differences or vertical distances between the actual and 
# predicted values) versus the predicted values to diagnose our regression model.
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()


from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % ( mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
# MSE train: 19.958, test: 27.196

# More MSE on test data means model is overfitted

# R2 is better for interpret the performance of the model
# it is  1 - (SSE/SST) here SST is the var(y) or total sum of squares
# It lies b/w 0 to 1 for training data and sometimes -ve for test data
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),
                                      r2_score(y_test, y_test_pred)))
# R^2 train: 0.765, test: 0.673
# R^2 also indicates the overfitting

# Ridge uses the L2 penalization.
# Least Absolute Shrinkage and Selection Operator (LASSO) uses L1 penalization.
# Elastic Net has an L1 penalty to generate sparsity and an L2 penalty to overcome some of the limitations of
# LASSO, such as the number of selected variables
# In all case we don't regularize intercept or w[0]
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.)

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)

from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)






# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:42:30 2019

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
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df=pd.read_csv(StringIO(csv_data))
print(df)
# StringIO allows us to read the string assigned to csv_data into a pandas DataFrame as if it was a regular CSV file on our hard drive.
print(df.isnull().sum())
print(df.dropna(axis=1))
# only drop rows where all columns are NaN
print(df.dropna(how='all'))
# drop rows that have less than 4 real values
print(df.dropna(thresh=4))
# only drop rows where NaN appear in specific columns (here: 'C')
print(df.dropna(subset=['B']))

imp=Imputer(missing_values='NaN',strategy='mean',axis=1)
imp.fit(df)
da=imp.transform(df)
print(da)

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],['red', 'L', 13.5, 'class2'],['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

# Giving numerical value of ordinal feature size
maping={'M':1,'L':2,'XL':3}
df['size']=df['size'].map(maping)
print(df)
# Reverting the values back to ordinal feature
inv_map={v:k for k,v in maping.items()}
print(inv_map)
df['size']=df['size'].map(inv_map)
print(df)

df['size']=df['size'].map(maping)
# Giving classes a numerical value
class_values={k:v for v,k in enumerate(np.unique(df['classlabel']))}
df['classlabel']=df['classlabel'].map(class_values)
# Similarly it can also be revert back like the size
print(df)
print(class_values)
inv_class={k:v  for v,k in class_values.items()}
print(inv_class)
df['classlabel']=df['classlabel'].map(inv_class)
print(df)

# Using LabelEncoder to transform class values
le=LabelEncoder()
df['classlabel']=le.fit_transform(df['classlabel'])
print(df)
# Reverting the values of class
df['classlabel']=le.inverse_transform(df['classlabel'])
print(df)

# Using LabelEncoder on color Feature
X=df[['color','size','price']]
print(X)
X['color']=le.fit_transform(X['color'])
print(X)
# Inversing
X['color']=le.inverse_transform(X['color'])
print(X)
X['color']=le.fit_transform(X['color'])
# If we give value to the color variables it can be problematic because no color can be larger or smaller than other one

# Now use OneHotEncoder
# It makes a single column of each value in the categorical feature with the value corresponding =1
# It gives a sparse matrix so we need to give sparse=False otherwise apply .toarray() after the transformation
# Categorical_features=[0] means transform 0 column 
# It is necessary because by default OneHotEncooder can transform numeric features also
one=OneHotEncoder(sparse=False,categorical_features=[0])
X=np.array(X)
X=one.fit_transform(X)
print(X)
# To reduce the correlation among variables, we can simply remove one feature column from the one-hot encoded array.
# By this matrxix computation gets easier and removing 1 column doesn't make any difference
X=X[:,1:]
print(X)

# Using get_dummies function of pandas 
df=pd.get_dummies(df[['color','size','price']],drop_first=True)
print(df)

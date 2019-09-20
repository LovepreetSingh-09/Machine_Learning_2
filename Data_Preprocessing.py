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

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:14:42 2019

@author: user
"""

import pyprind
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import pickle

df=pd.DataFrame()
basepath='aclImdb'
labels={'pos':1,'neg':0}
pb=pyprind.ProgBar(50000)
for s in ['train','test']:
    for l in labels:
        path=os.path.join(basepath,s,l)
        for files in os.listdir(path):
            with open(os.path.join(path,files),'r',encoding='utf-8') as infile:
                text=infile.read()
                df=df.append([[text,labels[l]]],ignore_index=True)
                pb.update()

df.columns=['review','sentiment']
df=df.reindex(np.random.permutation[df.index])
df.to_csv('movie_data.csv',encoding='utf-8',index=False)

df=pd.read_csv('movie_data.csv')
print(df.head(),'\n',len(df.index))
print(df['sentiment'][:5])

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(max_df=0.1,max_features=5000,stop_words='english')
help(CountVectorizer)
help(LDA)
lda=LDA(n_components=10,learning_method='batch',random_state=123)
X=count.fit_transform(df['review'].values)
X_topics=lda.fit_transform(X)
print(X_topics.shape) # (50000, 10)

# Components are matrix of topics X words/features
print(lda.components_.shape) # (10, 5000)

n_top_words=5
features=count.get_feature_names()
print(len(features)) # 5000
for idx,topics in enumerate(lda.components_):
    print('Topic : ',(idx+1))
    print([features[i] for i in topics.argsort()[-n_top_words:]])
    
b=np.array([1,3,5,7,2,9,4])
print(b)
print(b.argsort()[-5:])

family=X_topics[:,2].argsort()[::-1]
for i,f in enumerate(family[:2]):
    print('Movie : ',i+1,'\n') 
    print(df['review'][f][:350],'.....')
    

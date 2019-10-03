# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:13:46 2019

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
df=df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',encoding='utf-8',index=False)

df=pd.read_csv('movie_data.csv')
print(df.head(),'\n',len(df.index))
print(df['sentiment'][:5])

from nltk.corpus import stopwords
stop=stopwords.words('english')

def tokenizer(docs):
    docs=re.sub('<[^>]*>','',docs)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', docs)
    docs=re.sub('[\W]+',' ',docs.lower())+' '.join(emoticons).replace('-',' ')
    docs=[w for w in docs.split() if w not in stop]
    return docs

# Make a generator function for memory efficiency
def doc_stream(path):
    with open(os.path.join(path),'r',encoding='utf-8') as csv:
        # skip header file
        next(csv)
        for line in csv:
            text,label=line[:-3],int(line[-2])
            yield text, label

def minibatch(stream_doc,size):
    docs,l=[],[]
    try:
        for _ in range(size):
            text,label=next(stream_doc)
            docs.append(text)
            l.append(label)
    except StopIteration:
        return None,None
    return docs,l

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

vect=HashingVectorizer(tokenizer=tokenizer,n_features=2**21,decode_error='ignore')
clf=SGDClassifier(loss='log',random_state=1,n_iter=1)
doc=doc_stream('movie_data.csv')    
pb=pyprind.ProgBar(45)
classes=np.array([0,1])
for _ in range(45):
    X_train,y_train=minibatch(doc,size=1000)
    if not X_train:
        break
    X_train=vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes=classes)
    pb.update()
 
print(np.unique(y_train))
print(classes)

next(doc_stream(path='movie_data.csv'))

X_test,y_test=minibatch(doc,5000)
X_test=vect.transform(X_test)
print(clf.score(X_test,y_test)) # 0.870

clf.partial_fit(X_test,y_test,classes=classes)


dest=os.path.join('movie_classifier','pickle_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

print(os.getcwd())
pickle.dump(stop,open(os.path.join(dest,'stopwords.pkl'),'wb'),protocol=4)

pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'),protocol=4)


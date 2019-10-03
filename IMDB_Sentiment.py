# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:38:22 2019

@author: user
"""

import pyprind
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import pickle

print(os.getcwd())
basepath='aclImdb'
labels={'pos':1,'neg':0}
pb=pyprind.ProgBar(50000)
df=pd.DataFrame()
for s in ['train','test']:
    for l in ['pos','neg']:
        path=os.path.join(basepath,s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                text=infile.read()
                df=df.append([[text,labels[l]]],ignore_index=True)
                pb.update()
                
np.random.seed(1)
df.columns=['review','sentiment']      

df=df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',encoding='utf-8',index=False)
  
df.columns=['review','sentiment']      
df=pd.read_csv('movie_data.csv',encoding='utf-8')
print(df.head(),'\n',len(df.index))
print(df['sentiment'][:5])

from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(ngram_range=(1,2))
docs = np.array(['The sun is shining',
'The weather is sweet',
 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)

# A sparse matrix for saving memory
print(bag)
# An array with 0s
print(bag.toarray())

# Tf-idf vectorizer for assigning value to a word in a document on the basis of its presence across the document and the corpus
# It is the combination of Tf-idf transformer and countvectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# l2 normalization is by default
tfidf=TfidfVectorizer(norm='l2',use_idf=True,smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(docs).toarray())

# Cleaning the data :
print(df.loc[5,'review'][-50:])

def preprocessor(text):
    text=re.sub('<[^>]*>',' ',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # substituting or removing all non-word characters \W
    text=re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-',' ')
    return text

preprocessor("</a>This :) is :( a test :-)!")

df['review']=df['review'].apply(preprocessor)

# Spliting the sentence into single words by whitespace
def tokenizer(docs):
    return docs.split()

tokenizer('runners like running and thus they run')

# Stemming means converting a word into its base word like running to run
from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
def tokenizer_porter(docs):
    return [porter.stem(word) for word in docs.split()]

tokenizer_porter('runners like running and thus they run')

# Using stopwords:-
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')
print(stop[:50])
print([w for w in tokenizer_porter('runners like running and thus they run') if w not in stop])

# Training a model
X_train=df.loc[:25000,'review'].values
y_train=df.loc[:25000,'sentiment'].values
X_test=df.loc[25000:,'review'].values
y_test=df.loc[25000:,'sentiment'].values
print(X_train.shape)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

tfidf=TfidfVectorizer(lowercase=False,preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],'vect__stop_words': [stop, None],
'vect__tokenizer': [tokenizer, tokenizer_porter],
'clf__penalty': [ 'l2'],
'clf__C': [1.0, 10.0]},
 {'vect__ngram_range': [(1,1)],
'vect__stop_words': [stop, None],
'vect__tokenizer': [tokenizer,tokenizer_porter],
'vect__use_idf':[False],
'vect__norm':[None],
'clf__penalty': ['l1', 'l2'],
'clf__C': [1.0, 10.0]} ]

lr_tfidf=Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf=GridSearchCV(lr_tfidf,param_grid=param_grid,cv=5,n_jobs=5,verbose=1,scoring='accuracy')
gs_lr_tfidf.fit(X_train,y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)  # 0.899
clf = gs_lr_tfidf.best_estimator_
print(clf)
print('Test Accuracy: %.3f' % clf.score(X_test, y_test)) # 0.898



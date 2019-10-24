# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:56:16 2019

@author: user
"""
import torch
import sys
import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from keras.datasets import mnist
#
#(x_train,y_train),(x_test,y_test)=mnist.load_data()
#print(x_train.shape)
#print(x_test.shape)
#X_train=x_train/255.0
#X_test=x_test/255.0
#
#np.savez_compressed('mnist_scaled.npz',X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
mnist=np.load('mnist_scaled.npz')
mnist.files
X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)

fig, ax = plt.subplots(nrows=2, ncols=5,sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img=X_train[y_train==i][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=5,ncols=5, sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

class NeuralNet(object):
    def __init__(self,epochs=100,n_hidden=30,eta=0.001,l2=0,minibatch_size=1,shuffle=True,seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        
    def one_hot(self,y,n_classes):
        onehot=np.zeros((len(y),n_classes))
        for idx, val in enumerate(y.astype(int)):
            onehot[idx,val]=1
        return onehot
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-np.clip(z,-250,250)))
    
    def forward(self,X):
        z_h=np.dot(X,self.w_h)+self.b_h
        a_h=self.sigmoid(z_h)
        z_out=np.dot(a_h,self.w_out) + self.b_out
        a_out=self.sigmoid(z_out)
        return z_h,a_h,z_out,a_out
    
    def compute_cost(self,y_enc,output):
        l2_term=self.l2*(np.sum(self.w_h**2)+np.sum(self.w_out**2))
        term1=-y_enc*np.log(output)
        term2=(1-y_enc)*np.log(1-output)
        cost=np.sum(term1-term2)+l2_term
        return cost
    
    def predict(self,X):
        z_h,a_h,z_out,a_out=self.forward(X)
        y_pred=np.argmax(a_out,axis=1)
        return y_pred
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        n_classes=np.unique(y_train).shape[0]
        n_features=X_train.shape[1]
        self.b_h=np.zeros(self.n_hidden)
        self.w_h=self.random.normal(loc=0.0,scale=0.1,size=(n_features,self.n_hidden))
        self.b_out=np.zeros(n_classes)
        self.w_out=self.random.normal(loc=0.0,scale=0.1,size=(self.n_hidden,n_classes))
        epoch_strlen=len(str(self.epochs))
        y_train_enc=self.one_hot(y_train,n_classes)
        self.eval={'cost':[] , 'train_acc' : [] , 'valid_acc' : []}
        for i in range(self.epochs):
            indices=np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
            for idx in range(0,indices.shape[0]-self.minibatch_size+1,self.minibatch_size):
                batch_idx=indices[idx:idx+self.minibatch_size]
                z_h,a_h,z_out,a_out=self.forward(X_train[batch_idx])
                sigma_out=a_out-y_train_enc[batch_idx]
                sigmoid_derivative=a_h*(1-a_h)
                sigma_h=np.dot(sigma_out,self.w_out.T)*sigmoid_derivative
                grad_w_out=np.dot(a_h.T,sigma_out)
                grad_b_out=np.sum(sigma_out,axis=0)
                grad_w_h=np.dot(X_train[batch_idx].T,sigma_h)
                grad_b_h=np.sum(sigma_h,axis=0)
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out
            z_h, a_h, z_out, a_out = self.forward(X_train)
            cost = self.compute_cost(y_enc=y_train_enc,output=a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((np.sum(y_train ==y_train_pred)).astype(np.float) /X_train.shape[0])
            valid_acc = ((np.sum(y_valid ==y_valid_pred)).astype(np.float) /X_valid.shape[0])
            sys.stderr.write('\r%0*d/%d | Cost: %.2f ''| Train/Valid Acc.: %.2f%%/%.2f%% '%(epoch_strlen, i+1, self.epochs,cost,train_acc*100, valid_acc*100))
            sys.stderr.flush()  
            self.eval['cost'].append(cost)
            self.eval['train_acc'].append(train_acc)
            self.eval['valid_acc'].append(valid_acc)
        return self
    
nn = NeuralNet(n_hidden=100, l2=0.01,epochs=200,eta=0.0005, minibatch_size=100,shuffle=True, seed=1)
nn.fit(X_train=X_train[:55000],y_train=y_train[:55000],X_valid=X_train[55000:], y_valid=y_train[55000:])
                
plt.plot(range(nn.epochs), nn.eval['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()           

plt.plot(range(nn.epochs), nn.eval['train_acc'],label='training')
plt.plot(range(nn.epochs), nn.eval['valid_acc'], label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
print('Training accuracy: %.2f%%' % (acc * 100))
                




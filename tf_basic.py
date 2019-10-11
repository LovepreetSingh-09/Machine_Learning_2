# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:08:07 2019

@author: user
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.contrib.keras as keras

print(tf.__version__)
tf.compat.v1.disable_v2_behavior

g=tf.Graph()
with g.as_default():
    x=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None),name='x')
    w=tf.compat.v1.Variable(2.0,name='w')
    b=tf.compat.v1.Variable(0.7,name='b')
    z=w*x + b
    init=tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session(graph=g) as sess:
    sess.run(init)
    for t in [1.0,1.5,-1.4]:
        print('x=%4.1f --> z=%4.1f'%(t,sess.run(z,feed_dict={x:t})))
    

g=tf.Graph()
with g.as_default():
    x=tf.compat.v1.placeholder(dtype=tf.float32,shape=(None,5,3),name='X_input')
    x2=tf.reshape(x,shape=(-1,15),name='X_eshaped')
    xsum=tf.reduce_sum(x2,axis=0,name='Col_Sum')
    xmean=tf.reduce_mean(x2,name='Col_Mean')
with tf.compat.v1.Session(graph=g) as sess:
    x_array=np.arange(30).reshape(2,5,3)
    print(x_array)
    print(sess.run(x2,feed_dict={x:x_array}))
    print(sess.run(xsum,feed_dict={x:x_array}))
    print(sess.run(xmean,feed_dict={x:x_array}))
    
    
X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])
class TfLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01,
        random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        ## build the model
        with self.g.as_default():
            ## set graph-level random-seed
            tf1.set_random_seed(random_seed)
            self.build()
            ## create initializer
            self.init_op = tf1.global_variables_initializer()
    def build(self):
        ## define placeholders for inputs
        self.X = tf1.placeholder(dtype=tf.float32, shape=(None, self.x_dim), name='x_input')
        self.y = tf1.placeholder(dtype=tf.float32,shape=(None),name='y_input')
        print(self.X)
        print(self.y)
        ## define weight matrix and bias vector
        w = tf.Variable(tf.zeros(shape=(1)),name='weight')
        b = tf.Variable(tf.zeros(shape=(1)), name="bias")
        print(w)
        print(b)
        self.z_net = tf.squeeze(w*self.X + b,name='z_net')
        print(self.z_net)
        sqr_errors = tf.square(self.y - self.z_net, name='sqr_errors')
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')
        optimizer = tf1.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)

lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)

def train_linreg(sess, model, X_train, y_train, num_epochs=10):
    ## initialiaze all variables: W and b
    sess.run(model.init_op)
    training_costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost],feed_dict={model.X:X_train,model.y:y_train})
        training_costs.append(cost)
    return training_costs

sess = tf1.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, X_train, y_train)
print(training_costs)
plt.plot(range(1,len(training_costs) + 1), training_costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training Cost')
plt.show()

def predict_linreg(sess, model, X_test):
    y_pred = sess.run(model.z_net,feed_dict={model.X:X_test})
    return y_pred

plt.scatter(X_train, y_train,marker='s', s=50, label='Training Data')
plt.plot(range(X_train.shape[0]),predict_linreg(sess, lrmodel, X_train),color='gray', marker='o',
         markersize=6, linewidth=3,label='LinReg Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()

tf.RegisterGradient

from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
X_train=x_train.reshape(60000,784)
X_test=x_test.reshape(10000,784)

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_test
print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)
import keras
y_train_onehot = keras.utils.to_categorical(y_train)
print(y_train_onehot.shape)

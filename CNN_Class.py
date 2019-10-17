# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:45:09 2019

@author: user
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.keras as keras
from scipy.signal import convolve2d
from keras.datasets import mnist

(x_train,Y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
print(x_test.shape)
X_tr=x_train.reshape(60000,784)
X_test=x_test.reshape(10000,784)
X_train, y_train = X_tr[:50000,:], Y_train[:50000]
X_valid, y_valid = X_tr[50000:,:], Y_train[50000:];X_valid.shape
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = (X_valid - mean_vals)/std_val
print(X_valid.shape,y_valid.shape)
X_test_centered = (X_test - mean_vals)/std_val

def batch_generator(X, y, batch_size=64,shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])


class ConvNN(object):
    def __init__(self, batchsize=64,epochs=20, learning_rate=1e-4,dropout_rate=0.5,
                 shuffle=True, random_seed=None):
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        g = tf.Graph()
        with g.as_default():
        ## set random-seed:
            tf1.set_random_seed(random_seed)
            ## build the network:
            self.build()
            ## initializer
            self.init_op = tf1.global_variables_initializer()
            ## saver
            self.saver = tf1.train.Saver()
            ## create a session
        self.sess = tf1.Session(graph=g)
        
    def build(self):
        ## Placeholders for X and y:
        tf_x = tf1.placeholder(tf.float32, shape=[None, 784], name='tf_x')
        tf_y = tf1.placeholder(tf.int32, shape=[None], name='tf_y')
        is_train = tf1.placeholder(tf.bool,shape=(),name='is_train')
        ## reshape x to a 4D tensor:
        ## [batchsize, width, height, 1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],name='input_x_2dimages')
        ## One-hot encoding:
        tf_y_onehot = tf1.one_hot(indices=tf_y, depth=10,dtype=tf.float32, name='input_y_onehot')
        ## 1st layer: Conv_1
        h1 = tf1.layers.conv2d(tf_x_image, kernel_size=(5, 5),filters=32, activation=tf.nn.relu)
        ## MaxPooling
        h1_pool = tf1.layers.max_pooling2d(h1,pool_size=(2, 2),strides=(2, 2))
        ## 2n layer: Conv_2
        h2 = tf1.layers.conv2d(h1_pool, kernel_size=(5, 5),filters=64,activation=tf.nn.relu)
        ## MaxPooling
        h2_pool = tf1.layers.max_pooling2d(h2, pool_size=(2, 2), strides=(2, 2))
        ## 3rd layer: Fully Connected
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool,shape=[-1, n_input_units])
        h3 = tf1.layers.dense(h2_pool_flat, 1024,activation=tf.nn.relu)
        ## Dropout
        h3_drop = tf1.layers.dropout(h3,rate=self.dropout_rate, training=is_train)
        ## 4th layer: Fully Connected (linear activation)
        h4 = tf1.layers.dense(h3_drop, 10,activation=None)
        ## Prediction
        predictions = {'probabilities': tf.nn.softmax(h4, name='probabilities'),
                       'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels') }
        ## Loss Function and Optimization
        cross_entropy_loss = tf.reduce_mean(tf1.nn.softmax_cross_entropy_with_logits(
                logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')
        ## Optimizer:
        optimizer = tf1.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss,name='train_op')
        ## Finding accuracy
        correct_predictions = tf.equal( predictions['labels'], tf_y, name='correct_preds')
        accuracy = tf.reduce_mean(tf1.cast(correct_predictions, tf.float32), name='accuracy')
        print(accuracy)
        
    def save(self, epoch, path='./tflayers-model/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Saving model in %s' % path)
        self.saver.save(self.sess, os.path.join(path, 'model.ckpt'),global_step=epoch)
        
    def load(self, epoch, path):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess,os.path.join(path, 'model.ckpt-%d' % epoch))
    
    def train(self, training_set, validation_set=None,initialize=True):
        ## initialize variables
        if initialize:
            self.sess.run(self.init_op) 
        self.train_cost_ = []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])
        for epoch in range(1, self.epochs+1):
            batch_gen = batch_generator(X_data, y_data,shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_x,batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0': batch_x,'tf_y:0': batch_y, 'is_train:0': True} ## for dropout
                loss, _ = self.sess.run( ['cross_entropy_loss:0', 'train_op'],feed_dict=feed)
                avg_loss += loss
            print('Epoch %02d: Training Avg. Loss: ' '%7.3f' % (epoch, avg_loss), end=' ')
            if validation_set is not None:
                feed = {'tf_x:0': batch_x,'tf_y:0': batch_y,'is_train:0' : False} ## for dropout
                valid_acc = self.sess.run('accuracy:0', feed_dict=feed)
                print('Validation Acc: %7.3f' % valid_acc)
            else:
                print()
            
    def predict(self, X_test, return_proba=False):
        feed = {'tf_x:0' : X_test,'is_train:0' : False} ## for dropout
        if return_proba:
            return self.sess.run('probabilities:0',feed_dict=feed)
        else:
            return self.sess.run('labels:0', feed_dict=feed)


cnn = ConvNN(random_seed=123)
cnn.train(training_set=(X_train_centered, y_train),
          validation_set=(X_valid_centered, y_valid), initialize=True)
cnn.save(epoch=20)

del cnn

cnn2 = ConvNN(random_seed=123)
cnn2.load(epoch=20, path='./tflayers-model/')
print(cnn2.predict(X_test_centered[:10, :]))
preds = cnn2.predict(X_test_centered)
print('Test Accuracy: %.2f%%' % (100* np.sum(y_test == preds)/len(y_test)))











# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:38:43 2019

@author: user
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.keras as keras

mnist=np.load('mnist_scaled.npz')
files=mnist.files
print(files)
X_train,y_train,X_test,y_test=[mnist[i] for i in files]
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

n_classes=len(np.unique(y_train))
n_features=X_train.shape[1]
g=tf.Graph()
with g.as_default():
    tf_x=tf1.placeholder(shape=(None,n_features), dtype=tf.float64,name='tf_x')
    tf_y=tf1.placeholder(dtype=tf.int32,shape=None,name='tf_y')
    yone_hot=tf.one_hot(indices=tf_y, depth=n_classes)
    h1=tf1.layers.dense(inputs=tf_x,units=50,activation=tf1.tanh,name='layer_1')
    h2=tf1.layers.dense(inputs=h1,units=50, activation=tf1.tanh,name='layer_2')
    logits=tf1.layers.dense(inputs=h2,units=n_classes,activation=None,name='layers3')
    classes=tf1.argmax(logits,axis=1,name='classes')

with g.as_default():
    cost=tf1.losses.softmax_cross_entropy(logits=logits,onehot_labels=yone_hot)
    optim=tf1.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op=optim.minimize(loss=cost)
    init=tf1.global_variables_initializer()
    saver=tf1.train.Saver()

## run the variable initialization operator
n_epochs = 150
training_costs = []
with tf1.Session(graph=g) as sess:
    sess.run(init)
    # train the model for n_epochs
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op],feed_dict={tf_x: X_train,tf_y: y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))
    saver.save(sess, './trained-model')

g2=tf.Graph()
with tf1.Session(graph=g2) as sess:
# all of the information
# regarding the graph is saved as metadata in the file with the .meta extension.
    new=tf1.train.import_meta_graph('./trained-model.meta')
    print(new)
    new.restore(sess,'./trained-model')
    print(sess)
    y_pred=sess.run('classes:0',feed_dict={'tf_x:0':X_test})
    print(y_pred.shape)
    print(y_pred[:5])
    print(np.sum(y_test==y_pred)/10000)  
    
g = tf.Graph()
with g.as_default():
    arr = np.array([[1., 2., 3., 3.5],[4., 5., 6., 6.5],[7., 8., 9., 9.5]])
    T1 = tf.constant(arr, name='T1')
    print(T1)
    s = T1.get_shape()
    print('Shape of T1 is', s)
    T2 = tf.Variable(tf1.random_normal(shape=s))
    print(T2)
    T3 = tf.Variable(tf1.random_normal(shape=(s.as_list()[0],)))
    print(T3)    
    T4 = tf.reshape(T1, shape=[1, 1, -1], name='T4')
    print(T4)
    T5 = tf.reshape(T1, shape=[1, 3, -1],name='T5')
    print(T5)
    T6 = tf.transpose(T5, perm=[2, 1, 0],name='T6')
    print(T6)
    T7 = tf.transpose(T5, perm=[0, 2, 1],name='T7')
    print(T7)
    t5_splt = tf.split(T5,num_or_size_splits=2,axis=2, name='T8')
    print(t5_splt)
    t1 = tf.ones(shape=(5, 1),dtype=tf.float32, name='t1')
    t2 = tf.zeros(shape=(5, 1),dtype=tf.float32, name='t2')
    print(t1)
    print(t2)
    t3 = tf.concat([t1, t2], axis=0, name='t3')
    print(t3)
    t4 = tf.concat([t1, t2], axis=1, name='t4')
    print(t4)
    
with tf1.Session(graph = g) as sess:
    print(sess.run(T4))
    print()
    print(sess.run(T5))   
    print(t3.eval())
    print()
    print(t4.eval())




x, y = 1.0, 2.0
g = tf.Graph()
with g.as_default():
    tf_x = tf1.placeholder(dtype=tf.float32,shape=None, name='tf_x')
    tf_y = tf1.placeholder(dtype=tf.float32,shape=None, name='tf_y')
    res = tf.cond(tf_x < tf_y,
                  lambda: tf.add(tf_x, tf_y,name='result_add'),
                  lambda: tf.subtract(tf_x, tf_y, name='result_sub'))
    print('Object:', res)
    f1 = lambda: tf.constant(1)
    f2 = lambda: tf.constant(0)
    result = tf.case([(tf.less(x, y), f1)], default=f2)
    print(result)
    i = tf.constant(0)
    threshold = 100
    c = lambda i: tf.less(i, 100)
    b = lambda i: tf.add(i, 1)
    r = tf.while_loop(cond=c, body=b, loop_vars=[i])
    print(b)
        
with tf1.Session(graph=g) as sess:
    print('x < y: %s -> Result:' % (x < y),res.eval(feed_dict={'tf_x:0': x,'tf_y:0': y}))
    x, y = 2.0, 1.0
    print('x < y: %s -> Result:' % (x < y),res.eval(feed_dict={'tf_x:0': x,'tf_y:0': y}))
 

def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf1.get_variable(name='weights',shape=(data_shape[1],n_classes),dtype=tf.float32)
    bias = tf1.get_variable(name='bias',initializer=tf.zeros(shape=n_classes))
    logits = tf.add(tf.matmul(data, weights),bias,name='logits')
    return logits, tf.nn.softmax(logits)

def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf1.Variable( tf1.random_normal(shape=(data_shape[1],n_hidden)),name='w1')
    b1 = tf1.Variable(tf.zeros(shape=n_hidden),name='b1')
    hidden = tf.add(tf.matmul(data, w1), b1,name='hidden_pre-activation')
    hidden = tf.nn.relu(hidden, 'hidden_activation')
    w2 = tf1.Variable(tf1.random_normal(shape=(n_hidden,data_shape[1])),name='w2')
    b2 = tf1.Variable(tf.zeros(shape=data_shape[1]), name='b2')
    output = tf.add(tf.matmul(hidden, w2), b2,name = 'output')
    return output, tf.nn.sigmoid(output)
       
batch_size=64
g = tf.Graph()
with g.as_default():
    tf_X = tf1.placeholder(shape=(batch_size, 100), dtype=tf.float32,name='tf_X')
    ## build the generator
    with tf1.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X,n_hidden=50)
    with tf1.variable_scope('classifier') as scope:
        ## classifier for the original data:
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(shape=batch_size))
        ## reuse the classifier for generated data
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1], labels=tf.zeros(shape=batch_size))

with tf1.Session(graph=g) as sess:
    sess.run(tf1.global_variables_initializer())        
    file_wtiter=tf1.summary.FileWriter(logdir='./logs/',graph=g)

# tensorboard --logdir Documents\Spyder\logs

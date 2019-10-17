# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:55:42 2019

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

x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]
# same padding
print(np.convolve(x,w,mode='same'))

X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]
print(convolve2d(X,W,mode='same'))

from scipy.misc import imread
image=imread('./example-image.png')
print(image.shape)
print('Number of channels:', image.shape[2])
print('Image data type:', image.dtype)
print(image[100:103,100:103,:])

from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
print(x_test.shape)
X_train=x_train.reshape(60000,784)
X_test=x_test.reshape(10000,784)
X_train, y_train = X_train[:50000,:], y_train[:50000]
X_valid, y_valid = X_train[50000:,:], y_train[50000:]
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = (X_valid - mean_vals)/std_val
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

def conv_layer(input_tensor,name,kernel_size,n_outputs,strides=(1,1,1,1),padding_mode='SAME'):
    with tf1.variable_scope(name):
        # convolution_layer = [batch x height x width x depth/input_channels]
        n_channels=input_tensor.get_shape().as_list()
        n_channels=n_channels[-1]
        #weights = [batch x height x width x depth/input_channels x n_outputs]
        weight_shape=list(kernel_size) + [n_channels,n_outputs]
        weights=tf1.get_variable(shape=weight_shape,name='_weights')
        print(weights)
        biases=tf1.get_variable(initializer=tf.zeros(shape=[n_outputs]),name='biases')
        print(biases)
        conv=tf1.nn.conv2d(input=input_tensor,filter=weights,strides=strides,padding=padding_mode)
        print(conv)
        conv=tf.nn.bias_add(conv,biases,name='pre_activation')
        print(conv)
        conv=tf.nn.relu(conv,name='activation')
        return conv

g = tf.Graph()
with g.as_default():
    x = tf1.placeholder(tf.float32, shape=[None, 28, 28, 1])
    conv_layer(x, name='convtest',kernel_size=(3, 3),n_outputs=32)

del g, x

def fc_layer(input_tensor,n_outputs,name,activation_fn=None):
    with tf1.variable_scope(name):
        # Neglecting the no. of batches
        input_shape=input_tensor.get_shape().as_list()[1:]
        input_shape=np.prod(input_shape)
        if input_shape>1:
            input_tensor=tf.reshape(input_tensor,shape=(-1,input_shape))
            print(input_tensor)
        weight_shape=[input_shape,n_outputs]
        weights=tf1.get_variable(shape=weight_shape,name='_weights')
        print(weights)
        biases=tf1.get_variable(initializer=tf.zeros(shape=n_outputs),name='biases')
        print(biases)
        layer=tf.matmul(input_tensor,weights)
        print(layer)
        layer=tf.nn.bias_add(layer,biases,name='pre-activation')
        if activation_fn is None:
            return layer
        layer=activation_fn(layer,name='activation')
        print(layer)
        return layer

g = tf.Graph()
with g.as_default():
    x = tf1.placeholder(tf.float32,shape=[None, 28, 28, 1])
    fc_layer(x, name='fctest', n_outputs=32,activation_fn=tf.nn.relu)
del g, x

def build_cnn():
    ## Placeholders for X and y:
    tf_x = tf1.placeholder(tf.float32, shape=[None, 784],name='tf_x')
    tf_y = tf1.placeholder(tf.int32, shape=[None],name='tf_y')
    # reshape x to a 4D tensor:
    # [batchsize, width, height, 1]   
    tf_x_img=tf.reshape(tf_x,shape=(-1,28,28,1),name='images')
    y_onehot=tf.one_hot(indices=tf_y,depth=10,dtype=tf.float32,name='tf_one_hot')
    h1=conv_layer(tf_x_img,n_outputs=32,kernel_size=(5,5),padding_mode='VALID',name='conv_1')
    h1_pool=tf.nn.max_pool(h1,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
    h2=conv_layer(h1_pool,kernel_size=(5,5),n_outputs=64,padding_mode='VALID',name='conv_2')
    h2_pool=tf.nn.max_pool(h2,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
    h3=fc_layer(h2_pool,n_outputs=1024,activation_fn=tf.nn.relu,name='fc_1')
    keep_prob=tf1.placeholder(dtype=tf.float32,name='keep_prob')
    h3_drop=tf1.nn.dropout(h3,keep_prob=keep_prob,name='dropout_layer')
    h4=fc_layer(h3_drop,n_outputs=10,activation_fn=None,name='fc_2')
    predictions = {'probabilities': tf.nn.softmax(h4, name='probabilities'),
                   'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32,name='labels')}
    loss=tf.reduce_mean(tf1.nn.softmax_cross_entropy_with_logits(logits=h4,labels=y_onehot),name='loss_cross_entropy')
    optimizer=tf1.train.AdamOptimizer(learning_rate)
    optimizer=optimizer.minimize(loss,name='train_op')
    print(loss)
    correct_predictions=tf.equal(predictions['labels'],tf_y,name='correct_preds')
    accuracy=tf.reduce_mean(tf1.cast(correct_predictions,tf.float32),name='accuracy')
    print(accuracy)
 
def train(sess,training_set,validation_set=None,initializer=True,epochs=20,shuffle=True,dropout=0.5,random_seed=None):
    X_data=np.array(training_set[0])
    y_data=np.array(training_set[1])
    training_loss=[]
    if initializer:
        sess.run(tf1.global_variables_initializer())
    np.random.seed(random_seed)
    for epoch in range(1,epochs+1):
        batch_gen=batch_generator(X_data,y_data,shuffle=shuffle)
        avg_loss=0.0
        for i, (batch_x,batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x,'tf_y:0': batch_y,'keep_prob:0': dropout}
            loss, _ = sess.run(['loss_cross_entropy:0', 'train_op'],feed_dict=feed)
            avg_loss=avg_loss + loss
        training_loss.append(avg_loss)
        print('Epoch %02d Training Avg. Loss: %7.3f' % (epoch, avg_loss), end=' ')
        if validation_set is not None:
            feed={'tf_x:0':validation_set[0],'tf_y:0':validation_set[1],'keep_prob:0':1}
            valid_acc=sess.run('accuracy:0',feed_dict=feed)
            print('validation_accuracy : ',valid_acc)
        print()

def predict(sess,X_test,return_proba=False):
    feed={'tf_x:0':X_test,'keep_proba:0':1}
    if return_proba:
        return sess.run('probabilities:0',feed_dict=feed)
    else:
        return sess.run('labels:0',feed_dict=feed)

def save(sess,saver,epoch,path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving Model.........')
    saver.save(sess,os.path.join(path,'cnn-model.ckpt'),global_step=epoch)

def load(sess,epochs,saver,path):
    print('Loading Model............')
    saver.restore(sess,os.path.join(path,'cnn-model.ckpt-%d'%epochs))


learning_rate=1e-04
random_seed=123

g=tf.Graph()
with g.as_default():
    tf1.set_random_seed(random_seed)
    build_cnn()
    saver=tf1.train.Saver()

with tf1.Session(graph=g) as sess:
    train(sess,training_set=[X_train_centered,y_train],validation_set=[X_valid_centered,
          y_valid],epochs=20,random_seed=random_seed,initializer=True,shuffle=True)
    save(sess,saver,epoch=20)

del g
    
config.gpu_options.allow_growth = True
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

g2 = tf.Graph()
with g2.as_default():
    tf1.set_random_seed(random_seed)
    ## build the graph
    build_cnn()
    saver = tf.train.Saver()

with tf1.Session(graph=g2) as sess:
    load(saver, sess,epoch=20, path='./model/')
    preds = predict(sess, X_test_centered,return_proba=False)
    print('Test Accuracy: %.3f%%' % (100*np.sum(preds == y_test)/len(y_test)))

np.set_printoptions(precision=2, suppress=True)
with tf1.Session(graph=g2) as sess:
    load(saver, sess,epoch=20, path='./model/')
    print(predict(sess, X_test_centered[:10],return_proba=False))
    print(predict(sess, X_test_centered[:10],return_proba=True))


## continue training for 20 more epochs
## without re-initializing :: initialize=False
## create a new session
## and restore the model
with tf1.Session(graph=g2) as sess:
    load(saver, sess,epoch=20, path='./model/')
    train(sess,training_set=(X_train_centered, y_train),
          validation_set=(X_valid_centered, y_valid),initialize=False,
          epochs=20,random_seed=123)
    save(saver, sess, epoch=40, path='./model/')
    preds = predict(sess, X_test_centered,return_proba=False)
    print('Test Accuracy: %.3f%%' % (100*np.sum(preds == y_test)/len(y_test)))





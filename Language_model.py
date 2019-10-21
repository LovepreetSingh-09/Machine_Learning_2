# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:53:19 2019

@author: user
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.keras as keras
from collections import Counter
import pyprind
from string import punctuation

with open('shakespear.txt') as file:
    text=file.read()

print(text[:5])
chars=set(text)
print(chars);len(chars)
char2int={word:i for i,word in enumerate(chars)}; char2int
int2char={i:word for i,word in enumerate(chars)}; int2char
text_int=np.array([char2int[c] for c in text],dtype=np.int32)
print(text_int.shape)

def reshape_data(sequence, batch_size, num_steps):
    tot_batch_len=batch_size*num_steps
    num_batches=int(len(sequence)/tot_batch_len)
    b=num_batches*tot_batch_len
    if b+1 > len(sequence):
        num_batches-=1
    b=num_batches*tot_batch_len
    x=sequence[0:b]
    print(x.shape)
    y=sequence[1:b+1]
    x_split=np.split(x,batch_size)
    print(len(x_split))
    y_split=np.split(y,batch_size)
    x=np.stack(x_split);x.shape
    y=np.stack(y_split)
    return x,y

def create_batch_generator(data_x, data_y, num_steps):
    batch_size, tot_batch_length = data_x.shape
    num_batches = int(tot_batch_length/num_steps)
    for b in range(num_batches):
        yield (data_x[:, b*num_steps:(b+1)*num_steps],data_y[:, b*num_steps:(b+1)*num_steps])
    

def get_top_char(probas, char_size, top_n=5):
    p = np.squeeze(probas)
    p[np.argsort(p)[:-top_n]] = 0.0
    p = p / np.sum(p)
    ch_id = np.random.choice(char_size, 1, p=p)[0]
    return ch_id

class CharRNN(object):
    def __init__(self, num_classes, batch_size=64, num_steps=100, lstm_size=128,
                 num_layers=1, learning_rate=0.001,keep_prob=0.5, grad_clip=5,sampling=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build(sampling=sampling)
            self.saver = tf1.train.Saver()
            self.init_op = tf1.global_variables_initializer()
    
    def build(self,sampling):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size = self.batch_size
            num_steps = self.num_steps
        tf_x = tf1.placeholder(tf.int32,shape=[batch_size, num_steps], name='tf_x')
        tf_y = tf1.placeholder(tf.int32,shape=[batch_size, num_steps],name='tf_y')
        tf_keepprob = tf1.placeholder(tf.float32,name='tf_keepprob')
        # One-hot encoding:
        x_onehot = tf.one_hot(tf_x, depth=self.num_classes)
        y_onehot = tf.one_hot(tf_y, depth=self.num_classes)
        ### Build the multi-layer RNN cells
        cells = tf1.nn.rnn_cell.MultiRNNCell([tf1.nn.rnn_cell.DropoutWrapper(
                tf1.nn.rnn_cell.BasicLSTMCell(self.lstm_size),
                output_keep_prob=tf_keepprob)for _ in range(self.num_layers)])
        self.initial_state=cells.zero_state(batch_size,tf.float32)
        lstm_outputs,self.final_state=tf1.nn.dynamic_rnn(cells,x_onehot,initial_state=self.initial_state)
        # [batch_size x num_steps x lstm_size]
        print(lstm_outputs)
        seq_reshaped=tf.reshape(lstm_outputs,shape=(-1,self.lstm_size),name='seq_reshaped')
        logits=tf1.layers.dense(inputs=seq_reshaped,units=self.num_classes,activation=None,name='logits')
        proba=tf.nn.softmax(logits,name='probabilities')
        y_reshaped = tf.reshape(y_onehot,shape=[-1, self.num_classes],name='y_reshaped')
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped),name='cost')
        tvars=tf1.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),self.grad_clip)
        optimizer=tf1.train.AdamOptimizer(self.learning_rate)
        train_op=optimizer.apply_gradients(zip(grads,tvars),name='train_op')

    def train(self, train_x, train_y,num_epochs, ckpt_dir='./model/'):
## Create the checkpoint directory
## if it does not exists
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        with tf1.Session(graph=self.g) as sess:
            sess.run(self.init_op)
            n_batches = int(train_x.shape[1]/self.num_steps)
            iterations = n_batches * num_epochs
            for epoch in range(num_epochs):
                # Train network
                new_state = sess.run(self.initial_state)
                loss = 0
                ## Mini-batch generator:
                bgen = create_batch_generator(train_x, train_y, self.num_steps)
                for b, (batch_x, batch_y) in enumerate(bgen):
                    iteration = epoch*n_batches + b
                    feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y,'tf_keepprob:0' : self.keep_prob,self.initial_state : new_state}
                    batch_cost, _, new_state = sess.run( ['cost:0', 'train_op',self.final_state],feed_dict=feed)
                    if iteration % 10 == 0:
                        print('Epoch %d/%d Iteration %d' '| Training loss: %.4f' % ( epoch + 1, num_epochs, iteration, batch_cost))
                ## Save the trained model
                self.saver.save( sess, os.path.join(ckpt_dir, 'language_modeling.ckpt'))        
        
    def sample(self, output_length,ckpt_dir, starter_seq="The "):
        observed_seq = [ch for ch in starter_seq]
        with tf1.Session(graph=self.g) as sess:
            self.saver.restore(sess,tf1.train.latest_checkpoint(ckpt_dir))
            ## 1: run the model using the starter sequence
            new_state = sess.run(self.initial_state)
            for ch in starter_seq:
                x = np.zeros((1, 1))
                x[0, 0] = char2int[ch]
                feed = {'tf_x:0': x,'tf_keepprob:0': 1.0, self.initial_state: new_state}
                proba, new_state = sess.run(['probabilities:0', self.final_state],feed_dict=feed)
            ch_id = get_top_char(proba, len(chars))
            observed_seq.append(int2char[ch_id])
            ## 2: run the model using the updated observed_seq
            for i in range(output_length):
                x[0,0] = ch_id
                feed = {'tf_x:0': x,'tf_keepprob:0': 1.0,self.initial_state: new_state}
                proba, new_state = sess.run(['probabilities:0', self.final_state],feed_dict=feed)
                ch_id = get_top_char(proba, len(chars))
                observed_seq.append(int2char[ch_id])
        return ''.join(observed_seq)  

batch_size = 64
num_steps = 100
train_x, train_y = reshape_data(text_int,batch_size,num_steps)
train_x.shape

rnn = CharRNN(num_classes=len(chars), batch_size=batch_size)
rnn.train(train_x, train_y,num_epochs=100,ckpt_dir='./model-100/')

del rnn
np.random.seed(123)
rnn = CharRNN(len(chars), sampling=True)
print(rnn.sample(ckpt_dir='./model-100/',output_length=500))            

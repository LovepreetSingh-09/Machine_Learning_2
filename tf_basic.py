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
import tensorflow.keras as keras

print(tf.__version__)

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
print(y_train_onehot.shape)

n_features = X_train_centered.shape[1]
n_classes = 10
random_seed = 123
np.random.seed(random_seed)
g = tf.Graph()
with g.as_default():
    tf1.set_random_seed(random_seed)
    tf_x = tf1.placeholder(dtype=tf.float32,shape=(None, n_features), name='tf_x')
    tf_y = tf1.placeholder(dtype=tf.int32,shape=None, name='tf_y')
    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)
    h1 = tf1.layers.dense(inputs=tf_x, units=50, activation=tf.tanh,name='layer1')
    h2 = tf1.layers.dense(inputs=h1, units=50, activation=tf.tanh,name='layer2')
    logits = tf1.layers.dense(inputs=h2, units=10,activation=None,name='layer3')
    predictions = {'classes' : tf.argmax(logits, axis=1,name='predicted_classes'),
                   'probabilities' : tf.nn.softmax(logits, name='softmax_tensor')}
with g.as_default():
    cost = tf1.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
    optimizer = tf1.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=cost)
    init_op = tf1.global_variables_initializer()

def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)
    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)
    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])

sess = tf1.Session(graph=g)
## run the variable initialization operator
sess.run(init_op)
for epoch in range(50):
    training_costs = []
    batch_generator = create_batch_generator(X_train_centered, y_train,batch_size=64, shuffle=True)
    for batch_X, batch_y in batch_generator:
         ## prepare a dict to feed data to our network:
        feed = {tf_x:batch_X, tf_y:batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print(' -- Epoch %2d ''Avg. Training Loss: %.4f' % (epoch+1, np.mean(training_costs)))
        
feed = {tf_x : X_test_centered}
y_pred = sess.run(predictions['classes'],feed_dict=feed)

print('Test Accuracy: %.2f%%' % (100*np.sum(y_pred == y_test)/y_test.shape[0]))       

y_train_onehot = keras.utils.to_categorical(y_train)

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=50,input_dim=X_train_centered.shape[1],kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',activation='tanh'))
model.add(keras.layers.Dense(units=50,input_dim=50,kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',activation='tanh'))
model.add(keras.layers.Dense(units=y_train_onehot.shape[1],input_dim=50,kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',activation='softmax'))
# Weight decay is the regularization term
# Parameter that accelerates SGD in the relevant direction and dampens oscillations.
sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(optimizer=sgd_optimizer,loss='categorical_crossentropy')

model.fit(X_train_centered, y_train_onehot,batch_size=64, epochs=50,verbose=1,validation_split=0.1)

# By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
# verbose=0 will show you nothing (silent)
# verbose=1 will show you an animated progress bar 
# verbose=2 will just mention the number of epochy_train_pred = model.predict_classes(X_train_centered, verbose=0)
y_train_pred = model.predict_classes(X_train_centered, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])

correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100)) #  98.89%
y_test_pred = model.predict_classes(X_test_centered,verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100)) # 96.14 %


g = tf.Graph()
## define the computation graph
with g.as_default():
    ## define tensors t1, t2, t3
    t1 = tf.constant(np.pi)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2], [3, 4]])
     ## get their ranks
    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)
    ## get their shapes
    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    print('Shapes:', s1, s2, s3) # () (4,) (2, 2)
    print(r1) # Tensor("Rank:0", shape=(), dtype=int32)

# eval() can be done only in a session    
with tf1.Session(graph=g) as sess:
    print('Ranks:',r1.eval(), r2.eval(), r3.eval())
    # 0 1 2


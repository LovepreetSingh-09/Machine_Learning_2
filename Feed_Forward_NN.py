# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:56:16 2019

@author: user
"""

import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

#(x_train,y_train),(x_test,y_test)=mnist.load_data()
#print(x_train.shape)
#print(x_test.shape)
#X_train=x_train/255.0
#X_test=x_test/255.0

# np.savez_compressed('mnist_scaled.npz',X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
mnist=np.load('mnist_scaled.npz')
mnist.files
X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]

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
























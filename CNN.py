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









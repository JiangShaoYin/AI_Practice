#!   /usr/bin/python
#    coding:utf-8
#	 @file    opt_1.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-02 19:42:45
 
 
import tensorflow as tf
import numpy as np
BATCH_SIZE 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ =[[x1 + x2 +(rdm.rand()/10.0 - 0.05)] for (x1,x2) in X]

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.pla

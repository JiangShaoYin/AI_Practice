 
#		 coding:utf-8
#	 @file    opt_4_learing_rate.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-03 10:49:55
 
 
import tensorflow as tf
import numpy as np

#define loss function
w = tf.Variable(tf.constant(5, dtype = tf.float32))
loss = tf.square(w+1)

train_step = tf.train.ProximalGradientDescentOptimizer(1.0).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(w)
    for i in range(50):
        sess.run(train_step)
        print "after %d step,loss is %f,    w is %f" % (i,sess.run(loss),sess.run(w))

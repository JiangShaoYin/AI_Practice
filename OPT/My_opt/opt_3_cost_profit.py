#        coding:utf-8
#	 @file    opt_2_cost_profit.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-03 09:32:04
 
 
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 9
PROFIT = 1
#0
rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y = [[x1 + x2 + (rdm.rand()/10 -0.05)] for (x1, x2) in X]
print Y
#1
x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))
w1 = tf.Variable(tf.random_normal([2,1], stddev = 1, seed =1))
y = tf.matmul(x, w1)
#2 y > y_ stands for the predicated quantity is greater than actually needed,which will lead to loss 
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST*(y - y_), PROFIT * (y_ - y)))#if y > y_: do args1 else do args2
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#3
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = (i * BATCH_SIZE) % 32 + BATCH_SIZE 
        sess.run(train_step, feed_dict = {x : X[start : end], y_ : Y[start : end]})
        if i % 500 == 0:
            print "After runing %d step, w1 is :" % (i)
            print sess.run(w1)
    print "final w1 is :\n", sess.run(w1)

 
#		 coding:utf-8
#	 @file    opt_5_exponential_decay.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-03 11:13:06
 
 
import tensorflow as tf
import numpy as np

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1

w = tf.Variable(tf.constant(5, dtype = tf.float32))
loss = tf.square(w +1 )
#1
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                            global_step,
                                            LEARNING_RATE_STEP, 
                                            LEARNING_RATE_DECAY, 
                                            staircase = True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

#2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        sess.run(train_step)
        print "after %d step,  learning_rate is %f,    w is %f, loss is %f" % (i, sess.run(learning_rate), sess.run(w), sess.run(loss))

#    init_op=tf.global_variables_initializer()
#    sess.run(init_op)
#    for i in range(40):
#        sess.run(train_step)
#        learning_rate_val = sess.run(learning_rate)
#        global_step_val = sess.run(global_step)
#        w_val = sess.run(w)
     #   loss_val = sess.run(loss)
#        print "After %s steps: global_step is %f, w is %f, learning rate is %f, loss is %f" % (i, sess.run(global_step), sess.run(w), sess.run(learning_rate), sess.run(loss))

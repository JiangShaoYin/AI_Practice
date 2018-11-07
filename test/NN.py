#coding:utf-8
#0
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

random_num = np.random.RandomState(SEED)
print random_num
input_matrix = random_num.rand(32,2)
standard_answer = [[int(Input1 + Input2 < 1)] for (Input1,Input2) in input_matrix]
#print "input_matrix:\n",input_matrix
#print "Result:\n",standard_answer

#1 define forward propagation
input_layer = tf.placeholder(tf.float32, shape=(None, 2))
bp_answer = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

hidden_layer = tf.matmul(input_layer, w1)
computed_answer = tf.matmul(hidden_layer, w2)

#2 define loss and back propagation
loss = tf.reduce_mean(tf.square(computed_answer - bp_answer))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)    #minimize(loss):训练的方法，朝着loss最小的方向优化 
# 理解为一个标签，train_step == 后面的一大段函数

#3 train  STEPS times
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())#直到我们调用sess.run，变量都是未被初始化的
	print "w1,w2 before train:"
	print "w1:\n", sess.run(w1)
	print "w2:\n", sess.run(w2)

	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE    #train 8 element once
		sess.run(train_step, feed_dict = {input_layer : input_matrix[start : end], 
                                                        bp_answer : standard_answer[start : end]})#execute one training


	print "\n"
	print "w1:\n", sess.run(w1)
	print "w2:\n", sess.run(w2)

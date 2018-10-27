#0
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

random_num = np.random.RandomState(SEED)
print random_num
input_matrix = random_num.rand(32,2)
result = [[int(Input1 + Input2 < 1)] for (Input1,Input2) in input_matrix]
print "input_matrix:\n",input_matrix
print "Result:\n",result
#1 
input_layer = tf.placeholder(tf.float32, shape=(None, 2))
back_propagation_result = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

hidden_layer = tf.matmul(input_layer, w1)
output_layer = tf.matmul(hidden_layer, w2)

#2 define loss and back propagation
loss = tf.reduce_mean(tf.square(output_layer - back_propagation_result))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#3 train  STEPS times
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print "w1,w2 before train:"
	print "w1:\n", sess.run(w1)
	print "w2:\n", sess.run(w2)

	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict = {input_layer:input_matrix[start:end],back_propagation_result:result[start:end]})

	print "\n"

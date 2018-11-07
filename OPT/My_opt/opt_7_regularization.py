#coding:utf-8
#	 @file    opt_7_regularization.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-10-30 11:57:57
 
 
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
BATCH_SIZE = 30
SEED = 2

rdm = np.random.RandomState(SEED)
input_matrix = rdm.randn(300,2)
standard_answer = [int(x0*x0 + x1*x1) < 2 for(x0, x1) in input_matrix]
standard_answer_color = [['red' if y==1 else 'blue'] for y in standard_answer]
#formatting the matrix
input_matrix = np.vstack(input_matrix).reshape(-1,2)#the first args equal -1 which means it will be set by the second args automatically
standard_answer = np.vstack(standard_answer).reshape(-1,1)
#input_matrix[:,0] stand for input_matrix中的第一列元素
plt.scatter(input_matrix[:,0], input_matrix[:, 1], c = np.squeeze(standard_answer_color))
plt.show()

#1 define forward propagation
def get_weight(shape_formal_arge, regularizer):
        w = tf.Variable(tf.random_normal(shape_formal_arge))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

def get_bias(shape_formal_args):
    b = tf.Variable(tf.constant(0.01, shape = shape_formal_args))
    return b
input_layer = tf.placeholder(tf.float32, shape = (None,2))
bp_answer = tf.placeholder(tf.float32, shape = (None, 1))

w1 = get_weight([2,11], 0.01)
b1 = get_bias([11]) #一维11列的数组
hidden_layer = tf.nn.relu(tf.matmul(input_layer, w1) + b1)

w2 = get_weight([11,1], 0.01)
b2 = get_bias([1])
computed_answer = tf.matmul(hidden_layer, w2) + b2

#2 define loss function 
loss_mse = tf.reduce_mean(tf.square(computed_answer - bp_answer))
loss_regularization = loss_mse + tf.add_n(tf.get_collection('losses'))

#3.1 define back propagation(without regularization)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={input_layer : input_matrix[start : end],
                                        bp_answer : standard_answer[start : end]})
        if i % 2000 == 0:
            loss_mse_value = sess.run(loss_mse, feed_dict={input_layer : input_matrix,
                                      bp_answer : standard_answer})
            print("after %d steps,loss is : %f"%(i, loss_mse_value))
    xx, yy = np.mgrid[-3 : 3 : 0.01, -3 : 3 : 0.01]
    sgrid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(computed_answer, feed_dict = {input_layer : sgrid})
    probs = probs.reshape(xx.shape)
    print "w1:\n",sess.run(w1)
    print "b1:\n",sess.run(b1)
    print "w2:\n",sess.run(w2)
    print "b2:\n",sess.run(b2)

plt.scatter(input_matrix[:,0], input_matrix[:,1], c=np.squeeze(standard_answer))
#：告知 x、y 坐标和各点高度，用 levels 指定高度的点描上颜色 
plt.contour(xx, yy, probs, levels =[0.5])
plt.show()

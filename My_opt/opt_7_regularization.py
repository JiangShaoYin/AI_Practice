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
plt.scatter(input_matrix[:,0], input_matrix[:, 1], color = np.squeeze(standard_answer_color))
plt.show()

#1 define forward propagation
def get_weight(shape_formal_arge, regularizer)
        w = tf.Variable(tf.random_normal(shape_formal_arge), dtype = tf.float32)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)
        return w

def get_bias(shape_formal_arge)
    b = tf.Varible(tf.constant(0.01, shape = shape_formal_arge))
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



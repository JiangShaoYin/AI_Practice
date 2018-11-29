 
#	  coding:utf-8
#	 @file    forward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 21:35:21
 
 
import tensorflow as tf
import numpy as np

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32   #depth of convolution layer1
CONV2_SIZE =5
CONV2_KERNEL_NUM = 64   #depth of convolution layer1
FC_SIZE = 512
OUTPUT_NODE = 10

#private function ,only called by function forward()
def get_weight(shape, regularizer):
    #define variable w conform to normal distribution with standard deviation 0.1
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    #if the regularizatipn method is in use, regularize the args w in way of L2
    if regularizer!= None:
        #add regularizer outcome to collection losses
        tf.add_to_collection('losses',
                            tf.contrib.layers.l2_regularizer(regularizer)(w))
    #return the value of weight
    return w 

#get the value of the bais argument b which is a zero setting matrix
def get_bais(shape):
    #according to the shape of the metrix,return the bais value
    b = tf.Variable(tf.zeros(shape))
    #return the bais value   
    return b

#convolution compute
def conv2d(x, w):
    return tf.nn.conv2d(x,
            w,
            strides = [1,1,1,1], 
            padding = 'SAME')#padding == SAME means convolution lay equiped with 0

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


#define forward propagation method,do function forward(), 
#and return the computed result of the  nerual network
def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS ,CONV1_KERNEL_NUM], regularizer)#kernel_num is the depth of the convolution layer
    conv1_b = get_bais([CONV1_KERNEL_NUM])  #the order of b == the depth of convolution layer
    conv1 = conv2d(x, conv1_w)              #conv1_w is a matrix 28*28*1*32
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) #add bias to conv1
    pool1 = max_pool_2x2(relu1) #pooled the outcome of convolution layer1

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)#kernel_num is the depth of the convolution layer
    conv2_b = get_bais([CONV2_KERNEL_NUM])  #the order of b == the depth of convolution layer
    conv2 = conv2d(pool1, conv2_w)              #conv1_w is a matrix 28*28*1*32
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshape = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bais([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w) + fc1_b)
    if train:#如果是训练阶段，则将上一轮的输出fc1，随机舍去一定比例的计算结果（神经元）
        fc1 = tf.nn.dropout(fc1, 0.5)

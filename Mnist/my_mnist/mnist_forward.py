 
#	  coding:utf-8
#	 @file    forward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 21:35:21
 
 
import tensorflow as tf
import numpy as np
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

#private function ,only called by function forward()
def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    #if the regularizatipn method is in use, regularize the args w in way of L2
    if regularizer!= None:
        tf.add_to_collection('losses',
                            tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w 

#get the value of the bais args b which is a zero setting matrix
def get_bais(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

#define forward propagation method,do function forward(), 
#and return the computed result of the  nerual network
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE,LAYER1_NODE], regularizer)
    b1 = get_bais([LAYER1_NODE])
    #hidden layer1,call nonlinear activation function relu() to make the hidden layer1 nonlinear 
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bais([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y

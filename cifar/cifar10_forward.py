 
#	  coding:utf-8
#	 @file    forward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 21:35:21
 
 
import tensorflow as tf
import numpy as np
#input node width 3072 pixels
INPUT_NODE = 3072
#output node which have 10 possibility(0~9)
OUTPUT_NODE = 10
#hidden layer node quantity 
LAYER1_NODE = 500

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

#define forward propagation method,do function forward(), 
#and return the computed result of the  nerual network
def forward(x, regularizer):
    #define parameter w1 with shape[],regularizer option as the 2rd argument
    w1 = get_weight([INPUT_NODE,LAYER1_NODE], regularizer)
    #define bais argument 
    b1 = get_bais([LAYER1_NODE])
    #hidden layer1,call nonlinear activation function relu() to make the hidden layer1 nonlinear 
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    #the second hidden layer parameters 
    #define parameter w1 with shape[],regularizer option as the 2rd argument
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    #define bais argument 
    b2 = get_bais([OUTPUT_NODE])
    #define the outcome of the neural network
    y = tf.matmul(y1, w2) + b2
    #return the computed outcome of NN model
    return y

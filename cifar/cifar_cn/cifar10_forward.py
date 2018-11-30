#	  coding:utf-8
#	 @file    forward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 21:35:21
 
 
import tensorflow as tf
import numpy as np
INPUT_NODE = 3072  #1个图片有3072个像素点32*32*3
OUTPUT_NODE = 10   #输出层最后输出1*10矩阵
LAYER1_NODE = 500 #隐藏层的节点数量（神经元个数）

def get_weight(shape, regularizer):                         #获取参数w
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1)) #生成shape形状的参数矩阵w，其值服从平均值和标准偏差的正态分布，如果生成的值大于平均值2倍标准偏差，丢弃该值并重新选择。
    if regularizer!= None:                                  #如果启用正则化
        tf.add_to_collection('losses',                      #用l2正则化w，并将结果加入losses集合
                            tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w 

def get_bais(shape):                                         #生成偏执项b
    b = tf.Variable(tf.zeros(shape))                         #生成全shape形状的全0矩阵
    return b

def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE,LAYER1_NODE], regularizer) #生成一个参数矩阵，cifar中为3072 * 500（pic尺寸为1*3072）,隐藏层神经元个数为500 
    b1 = get_bais([LAYER1_NODE])                           #根据中间层神经元的个数，生成偏执项矩阵b1
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)                 #将神经元计算结果，过relu函数非线性激活，make the hidden layer1 nonlinear

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)    #第二个隐藏层
    b2 = get_bais([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2                             #过2层全连接层后的计算结果
    return y

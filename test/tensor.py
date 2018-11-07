 
#	  coding:utf-8
#	 @file    tensor.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-03 15:04:59
 
 
import tensorflow as tf
import numpy as np

a = tf.constant([1,2])
b = tf.constant([4,6])
print tf.add(a, b)

w = tf.Variable(tf.random_normal([2,3]))
print w

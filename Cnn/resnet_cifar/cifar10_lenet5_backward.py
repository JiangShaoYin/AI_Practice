#	  coding:utf-8
#	 @file    backward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 22:08:54
 
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets as nets
import cifar10_lenet5_generateds #1
import os
from tensorflow.examples.tutorials.mnist import input_data

LEARNING_RATE_BASE = 0.005		# 基础学习率
LEARNING_RATE_DECAY = 0.99		# 学习率的衰减率
REGULARIZER = 0.0001 			# 描述模型复杂度的正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99		# 滑动平均衰减率
BATCH_SIZE = 100          		# 一个训练batch中的训练数据个数

STEPS = 50000                           # 训练轮数
MODEL_SAVE_PATH = "./model"		# 模型存储路径
MODEL_NAME = "cifar10_model"    	# 模型命名
train_num_examples = 50000 		#2训练样本数

def backward():								#执行反向传播，训练参数w
    x = tf.placeholder(tf.float32, [None, 32,32,3])
    y_ = tf.placeholder(tf.float32, [None]) #
    
    y, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
    y = tf.reshape(y, shape=[-1, 10]) #预测值，相当于y

    global_step = tf.Variable(0, trainable = False)

    # 将label值进行onehot编码,10分类
    one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=10)
    # 定义损失函数和优化器
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=one_hot_labels))
    learning_rate = tf.train.exponential_decay(				# 设置指数衰减的学习率
    				               LEARNING_RATE_BASE,  	# 基础学习率，随着迭代的进行，更新变量时使用的学习率在此基础上递减 	
                                               global_step,			    # 当前迭代轮数
                                               train_num_examples / BATCH_SIZE,     # 过完所有训练数据所需迭代次数 	
                                               LEARNING_RATE_DECAY,		    # 指数学习衰减率
                                               staircase = True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)# 使用梯度下降优化损失函数，损失函数包含了交叉熵和正则化损失
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step) 				  # 初始化滑动平均类
    ema_op = ema.apply(tf.trainable_variables()) 																		  	# 对所有表示神经网络参数的变量进行滑动平均
    #正确率计算
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(one_hot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.control_dependencies([train_step, ema_op]):		        # 使用tf.control_dependencies机制一次完成多个操作。在此神经网络模型中，每过一遍数据既通过 
        train_op = tf.no_op(name = 'train')             		# 反向传播更新参数，又更新了每一个参数的滑动平均值
    saver = tf.train.Saver()

    img_batch, label_batch = cifar10_lenet5_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)#img_batch = [[32,32,3],[32,32,3]], label_batch=[[0,0,0,1,0...], [0,1,0....]]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state("./model")                 # 从"./model"中加载训练好的模型
        if ckpt and ckpt.model_checkpoint_path: 			# 若ckpt和保存的模型在指定路径中存在，则将保存的神经网络模型加载到当前会话中
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()                                 # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 启动QueueRunner,此时文件名队列已经进队  

        for i in range(STEPS):
            b_image, b_label = sess.run([img_batch, label_batch])

            _, step, accuracy_,  loss_,= sess.run([train_op, global_step, accuracy, loss], feed_dict={x: b_image,
                                                    y_: b_label})
            if i % 1000 == 0:
                print("after %d training step(s),accuracy : %g, loss is %g." % (step,accuracy_, loss_))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step = global_step)   # 保存当前模型，globle_step参数可以使每个被保存模型的文件名末尾都加上训练的轮数
        coord.request_stop()
        coord.join(threads)
def main():
    backward()

if __name__ == '__main__':                                                   #main function,
    main()


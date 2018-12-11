#	 coding:utf-8
#	 @file    cifar10_lenet5test.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-06 11:18:00
 
 
import tensorflow as tf
import numpy as np
import time
import cifar10_lenet5_backward
import cifar10_lenet5_generateds
import tensorflow.contrib.slim.nets as nets
from tensorflow.examples.tutorials.mnist import input_data
INTERVAL_TIME = 5
TEST_NUM = 100 #1
#test!

def test():
    with tf.Graph().as_default() as g:                                      #复现之前定义的计算图，并执行以下操作
        x = tf.placeholder(tf.float32, [None, 32,32,3])
        y_ = tf.placeholder(tf.float32, [None]) #
    
        y, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
        y = tf.reshape(y, shape=[-1, 10]) #预测值，相当于y
                
       # ema = tf.train.ExponentialMovingAverage(cifar10_lenet5_backward.MOVING_AVERAGE_DECAY)# 实现滑动平均模型，参数MOVING_AVERAGE_DECAY用于控制模型更新的速度，训练过程中会对每一个变量维护一个影子变量
       # ema_restore = ema.variables_to_restore()                                      # variable_to_restore()返回dict ({ema_variables : variables})，字典中保存变量的影子值和现值
       # saver = tf.train.Saver(ema_restore) 			                                    # 创建可还原滑动平均值的对象saver，测试时使用w的影子值，有更好的适配性
        saver = tf.train.Saver() 			                                    # 创建可还原滑动平均值的对象saver，测试时使用w的影子值，有更好的适配性


        one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=10)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(one_hot_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        img_batch,label_batch = cifar10_lenet5_generateds.get_tfrecord(TEST_NUM, isTrain=False)  #2 一次批获取 TEST_NUM 张图片和标签
        
        print "before true"
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(cifar10_lenet5_backward.MODEL_SAVE_PATH)    # 从指定路径中，加载训练好的模型
                if ckpt and ckpt.model_checkpoint_path:                                   # 若已有ckpt模型则执行以下恢复操作
                    saver.restore(sess, ckpt.model_checkpoint_path)                       # 恢复会话到当前的神经网络
     
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # 从ckpt.model_checkpoint_path中，通过字符 "/" 和 "-"提取出最后一个整数（保存的轮数），恢复轮数，

                    coord = tf.train.Coordinator()                                        #3开启线程协调器
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)        #4
                    xs,ys = sess.run([img_batch, label_batch])                            #5# 在 sess.run 中执行图片和标签的批获取
 
                    accuracy_score = sess.run(accuracy, # 计算准确率
                        feed_dict={x:xs, y_:ys})

                    print ("after %s training step(s), test accuracy = %g"
                            % (global_step, accuracy_score))

                    coord.request_stop()                                          #6
                    coord.join(threads)                                           #7

                else:                                                             #can not get checkpoint file ,print error infomation
                    print ("No checkpoint file found")
                    return
            time.sleep(INTERVAL_TIME)                                             # 设置等待时间，等backward生成新的checkpoint文件，再循环执行test函数

def main():
    test()
    
#main function,
if __name__ == '__main__':
    main()

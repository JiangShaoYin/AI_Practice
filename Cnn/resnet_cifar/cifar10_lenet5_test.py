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
BATCH_SIZE = 100
#test!

def test():
    with tf.Graph().as_default() as g:                                      #����֮ǰ����ļ���ͼ����ִ�����²���
        x = tf.placeholder(tf.float32, [None, 32,32,3])
        y_ = tf.placeholder(tf.float32, [None]) #
    
        y, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
        y = tf.reshape(y, shape=[-1, 10]) #Ԥ��ֵ���൱��y
                
        ema = tf.train.ExponentialMovingAverage(cifar10_lenet5_backward.MOVING_AVERAGE_DECAY)# ʵ�ֻ���ƽ��ģ�ͣ�����MOVING_AVERAGE_DECAY���ڿ���ģ�͸��µ��ٶȣ�ѵ�������л��ÿһ������ά��һ��Ӱ�ӱ���
        ema_restore = ema.variables_to_restore()                                      # variable_to_restore()����dict ({ema_variables : variables})���ֵ��б��������Ӱ��ֵ����ֵ
        saver = tf.train.Saver(ema_restore) 			                                    # �����ɻ�ԭ����ƽ��ֵ�Ķ���saver������ʱʹ��w��Ӱ��ֵ���и��õ�������


        one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=10)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(one_hot_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        img_batch,label_batch = cifar10_lenet5_generateds.get_tfrecord(TEST_NUM, isTrain=False)  #2 һ������ȡ TEST_NUM ��ͼƬ�ͱ�ǩ
        
        print "before true"
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(cifar10_lenet5_backward.MODEL_SAVE_PATH)    # ��ָ��·���У�����ѵ���õ�ģ��
                if ckpt and ckpt.model_checkpoint_path:                                   # ������ckptģ����ִ�����»ָ�����
                    saver.restore(sess, ckpt.model_checkpoint_path)                       # �ָ��Ự����ǰ��������
     
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # ��ckpt.model_checkpoint_path�У�ͨ���ַ� "/" �� "-"��ȡ�����һ����������������������ָ�������

                    coord = tf.train.Coordinator()                                        #3�����߳�Э����
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)        #4
                    xs,ys = sess.run([img_batch, label_batch])                            #5# �� sess.run ��ִ��ͼƬ�ͱ�ǩ������ȡ
 
                    accuracy_score = sess.run(accuracy, # ����׼ȷ��
                        feed_dict={x:xs, y_:ys})

                    print ("after %s training step(s), test accuracy = %g"
                            % (global_step, accuracy_score))

                    coord.request_stop()                                          #6
                    coord.join(threads)                                           #7

                else:                                                             #can not get checkpoint file ,print error infomation
                    print ("No checkpoint file found")
                    return
            time.sleep(INTERVAL_TIME)                                             # ���õȴ�ʱ�䣬��backward�����µ�checkpoint�ļ�����ѭ��ִ��test����

def main():
    test()
    
#main function,
if __name__ == '__main__':
    main()

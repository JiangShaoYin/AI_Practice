 
#	 coding:utf-8
#	 @file    cifar10_test.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-06 11:18:00
 
 
import tensorflow as tf
import numpy as np
import time
import cifar10_forward
import cifar10_backward
import cifar10_generateds
from tensorflow.examples.tutorials.mnist import input_data
INTERVAL_TIME = 5
TEST_NUM = 10000 #1
#test!
def test():
    #creates a new graph and places everything (declared inside its scope) into this graph.
    with tf.Graph().as_default() as g:
        #define placeholder x,which act as input image
        x = tf.placeholder(tf.float32, [None, cifar10_forward.INPUT_NODE])
        #define y as the computed result which will feed the parameter y_ below
        y_ = tf.placeholder(tf.float32,[None, cifar10_forward.OUTPUT_NODE])
        #execute forward propagation without regularization,and return it's outcome to y
        y = cifar10_forward.forward(x, None)

        #Create an ExponentialMovingAverage object ema
        ema = tf.train.ExponentialMovingAverage(cifar10_backward.MOVING_AVERAGE_DECAY)
        #method variable_to_restore() return a dict ({ema_variables : variables}) 
        ema_restore = ema.variables_to_restore()
        #Create a saver that loads variables from their saved shadow values.
        saver = tf.train.Saver(ema_restore)
        
        #if tf.argmax(y, 1) equals to tf.argmax(y_, 1),correct_prediction will be set True
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #cast the type of corrent_prediction from boolean to float and compute it's average
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        img_batch,label_batch = cifar10_generateds.get_tfrecord(TEST_NUM, isTrain=False)  #2
        
        #load trained model in the loop
        while True:
            #create session to manage the context
            with tf.Session() as sess:
                #fetch chechpoint from sepcified path
                ckpt = tf.train.get_checkpoint_state(cifar10_backward.MODEL_SAVE_PATH)
                #if got the checkpoint sucessfully do things below
                if ckpt and ckpt.model_checkpoint_path:
                    #restore the model to current neural network
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #fetch the step from the string ckpt.model_checkpoint and extract
                    #the last integer via charactor "/" & "-"
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()#3
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)#4
                    xs,ys = sess.run([img_batch, label_batch])#5

                    #compute accuracy_score via test data set
                    accuracy_score = sess.run(accuracy, 
                        feed_dict={x:xs, y_:ys})
                    #print the predict result
                    print ("after %s training step(s), test accuracy = %g"
                            % (global_step, accuracy_score))

                    coord.request_stop()#6
                    coord.join(threads)#7

                else:#can not get checkpoint file ,print error infomation
                    print ("No checkpoint file found")
                    #exit this moudle
                    return
            #set interval time to wait for the checkpoint file which the backward function produce 
            time.sleep(INTERVAL_TIME)

#define function main()
def main():
    #read test data from path "./data"
#   mnist = input_data.read_data_sets("./data", one_hot = True)
    #execute test functon
    test()
    
#main function,
if __name__ == '__main__':
    main()

 
#	 coding:utf-8
#	 @file    mnist_test.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-06 11:18:00
 
 
import tensorflow as tf
import numpy as np
import time
import mnist_forward
import mnist_backward
from tensorflow.examples.tutorials.mnist import input_data
INTERVAL_TIME = 3

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32,[None, mnist_forward.OUTPUT_NODE])
        #execute forward propagation without regularization,and return it's outcome to y
        y = mnist_forward.forward(x, None)
        #create ema with the predefined decay rate
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #cast the type of corrent_prediction from boolean to float and compute it's average
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                #fetch chechpoint from sepcified path
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #fetch the step from the string ckpt.model_checkpoint and extract
                    #the last integer via charactor "/" & "-"
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    #compute accuracy_score via test data set
                    accuracy_score = sess.run(accuracy, 
                        feed_dict={x:mnist.test.images, y_:mnist.test.labels})
                    print ("after %s training step(s), test accuracy = %g"
                            % (global_step, accuracy_score))
                else:
                    print ("No checkpoint file found")
                    return
            time.sleep(INTERVAL_TIME)
def main():
    mnist = input_data.read_data_sets("./data", one_hot = True)
    test(mnist)
if __name__ == '__main__':
    main()

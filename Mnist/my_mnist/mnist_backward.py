 
#	  coding:utf-8
#	 @file    backward.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-05 22:08:54
 
 
import tensorflow as tf
import numpy as np
import mnist_forward
import os
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "mnist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])#
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable = False)
    #cross entropy,to calculate the distance between the standard probability
    #distribution and the nerual network calculated probability distribution 
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,
                                            labels = tf.argmax(y_, 1))
    #compute average ce
    cem = tf.reduce_mean(ce)
    #compute total loss with cross entropy 
    loss = cem + tf.add_n(tf.get_collection('losses'))
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase = True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    #compute exponential moving average(ema) of all tainable variabels
        #create class ema,prepare to be computed
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        #aplly ema to all trainable variables
    ema_op = ema.apply(tf.trainable_variables())
    #bind operation train_step & ema_op together to realize two operations at time
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')
    #create class saver to save the session below
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #initialize all global variables
        sess.run(tf.global_variables_initializer())

        #restore module
        #fetch the checkpoint from path "./model"
        ckpt = tf.train.get_checkpoint_state("./model")
        #if checkpoint and it’s path exist,do restore()
        if ckpt and ckpt.model_checkpoint_path:
            #restore model to current neural network
            saver.restore(sess, ckpt.model_checkpoint_path)
        #end of restore

        for i in range(STEPS):
            #fetch dataset to be trained,assign train image to xs,train labels to ys
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            #calculate node train_op, loss,global_step and return the result to _,loss_value, step 
            #'_' means an anonymous variable which will not in use any more
            _, loss_value, step = sess.run([train_op, loss,global_step], 
                                            feed_dict = {x : xs, y_ : ys})
            #save current neural network with a 1000-step frequency,
            if i % 1000 == 0:
                print("after %d training step(s), loss on training batch is %g." % (step, loss_value))
                #save the global_step neural network(current NN)to specified path(which is MODEL_SAVE_PATH + MODEL_NAME)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step = global_step)
                
def main():
    mnist = input_data.read_data_sets("./data", one_hot = True)
    backward(mnist)

if __name__ == '__main__'
    main()































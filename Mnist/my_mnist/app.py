 
#	  coding:utf-8
#	 @file    app.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-07 16:16:16
 
 
import tensorflow as tf
import numpy as np
import mnist_forward
import mnist_backward
from PIL import Image

def pre_pic(picName):
    print picName
    img = Image.open("./pic/p8.png")
#    img = Image.open(picName)
    print '...........................................'
    reIm = img.resize((28,28),Image.ANTIALIAS)
    #transform the picture to greyscale(0 stands for pure black,255 stands for pure white
    #,the integers from 1~254 stand for grey level)
    im_array = np.array(reIm.convert('L'))
    print im_array
    #set threshold to distinguish between balck and white,if the vale > threshold,treat it as pure balck(255)
    threshold = 50
    for i in range(28):
        for j in range(28):
            #invert the image as the input images are just the opposite of images which the neural network training with
            im_array[i][j] = 255 - im_array[i][j]
            if (im_array[i][j] < threshold):
                im_array[i][j] = 0
            else:
                im_array[i][j] = 255
    print '...........................................'
    print im_array
#    print im_array
    #reshape the picture from 28*28 to 1*784
    reshaped_array = im_array.reshape([1,784])
    #cast the value from int to float
    reshaped_array = reshaped_array.astype(np.float32)
    #transfer the value from 0/255,to 0/1
    img_ready = np.multiply(reshaped_array, 1.0/255)
    return img_ready

def restore_model(testPicArr):
    with tf.Graph().as_default() as g:
        #compute forwa
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        #define the predicted value
        predictValue = tf.argmax(y,1)
        #define class ema to restore args that the backward function stored
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        #funtion variable_to_restore()
        ema_restore = ema.variables_to_restore()
        #Create a saver (an object with stored ema args )
        saver = tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                predictValue = sess.run(predictValue, feed_dict={x:testPicArr})
                return predictValue
            else:
                print "no checkpoint file found"
                return -1


def app():
    testNum = input("input the number of test pictures:")
    for i in range(testNum):
        #testPic = raw_input("./pic/" + "%d.png"(i))
        testPic = raw_input("./pic/2.png")
#    testPic = raw_input("./pic/1.png")
        testPicArr = pre_pic(testPic)
        predictValue = restore_model(testPicArr)
        print "the predict number is :%d"%predictValue
    print testPic
def main():
    app()

if __name__ == '__main__':
    main()

 
#	  coding:utf-8
#	 @file    app.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-07 16:16:16
 
 
import tensorflow as tf
import numpy as np
import mnist_forward
import mnist_backward
from PIL import Image
# set the number of the pictures to be tested
testNum = 10

#Image pretreatment.
def pre_pic(picName):
    #open the image by path which the formal parameter told
    img = Image.open(picName)
    #resize the test image to 28*28,and set antialias
    reIm = img.resize((28,28),Image.ANTIALIAS)
    #transform the picture to greyscale(0 stands for pure black,255 stands for pure white
    #,the integers from 1~254 stand for grey level)
    im_array = np.array(reIm.convert('L'))

    #set threshold to distinguish between balck and white,if the vale > threshold,treat it as pure balck(255)
    threshold = 50

    #set loop to traverse every pixel in the tested images
    for i in range(28):         #width 28
        for j in range(28):     #height 28
            #invert the image as the input images are just the opposite of images which the neural network training with
            im_array[i][j] = 255 - im_array[i][j]
            #if the vale < threshold,treat it as pure white(0)
            if (im_array[i][j] < threshold):
                im_array[i][j] = 0
            else:       #if the vale >= threshold,treat it as pure balck(255)
                im_array[i][j] = 255
    #reshape the picture from 28*28 to 1*784
    reshaped_array = im_array.reshape([1,784])
    #cast the value from int to float
    reshaped_array = reshaped_array.astype(np.float32)
    #transfer the value from 0/255,to 0/1
    img_ready = np.multiply(reshaped_array, 1.0/255)
    #return the prosessed  image
    return img_ready

#compute the tested picture and predict its number according restored model
def restore_model(testPicArr):
    #creates a new graph and places everything (declared inside its scope) into this graph.
    with tf.Graph().as_default() as g:
        #compute forwa
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        #define the predicted value
        predictValue = tf.argmax(y,1)

        #Create an ExponentialMovingAverage object ema
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        #method variable_to_restore() return a dict ({ema_variables : variables}) 
        ema_restore = ema.variables_to_restore()
        #Create a saver that loads variables from their saved shadow values.
        saver = tf.train.Saver(ema_restore)

        #create session to manage the context
        with tf.Session() as sess:
            #fetch chechpoint from sepcified path
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                #if got the checkpoint sucessfully do things below
            if ckpt and ckpt.model_checkpoint_path:
                    #restore the model to current neural network
                saver.restore(sess, ckpt.model_checkpoint_path)
                #input the testPicArr through feed_dict, and return the predicted answer to predictValue 
                predictValue = sess.run(predictValue, feed_dict={x:testPicArr})
                #return the value of the picture
                return predictValue
            else:
                #fail to fetch the checkpoint,print error infomation
                print "no checkpoint file found"
                #return -1 means this function ends abnormally
                return -1


def app():
    #test the pic in loop
    for i in range(testNum):
        #concatenate those strings below to create a new string testPic as the path of the tested picture
        testPic = "./pic/" + str(i) + ".jpg"
        #call function pre_pic() to execute image pretreatment
        testPicArr = pre_pic(testPic)
        #call function restore_model to discriminate the number in the tested picture
        predictValue = restore_model(testPicArr)
        #print the result
        print "the predict number is :%d"%predictValue

#define app() as the main function        
def main():
    app()

#execute main funcion
if __name__ == '__main__':
    main()

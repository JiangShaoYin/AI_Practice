 
#	  coding:utf-8
#	 @file    app.py
#	 @author  Sean(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-07 16:16:16
 
 
import tensorflow as tf
import numpy as np
from PIL import Image

def pre_pic(picName):
    print picName
    img = Image.open(picName)
    reIm = img.resizei((28,28),Image.ANTIALIAS)
    #transform the picture to greyscale(0 stands for pure black,255 stands for pure white
    #,the integers from 1~254 stand for grey level)
    im_array = np.array(reIm.convert('L'))
    #set threshold to distinguish between balck and white,if the vale > threshold,treat it as pure balck(255)
    threshold = 50
    for i in range(28):
        print picName
        for j in range(28):
            #invert the image as the input images are just the opposite of images which the neural network training with
            im_array[i][j] = 255 - im_arr[i][j]
            if (im_array[i][j] < threshold):
                im_array[i][j] = 0
            else:
                im_array[i][j] = 255


def application():
    testNum = input("input the number of test pictures:")
#    for i in range(testNum):
        #testPic = raw_input("./pic/" + "%d.png"(i))
    testPic = raw_input("./pic/1.png")
    pre_pic(testPic)
    print testPic

application()

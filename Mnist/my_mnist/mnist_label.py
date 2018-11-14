 
#        coding:utf-8
#	 @file    mnist_label.py
#	 @author  1801210547_江绍印(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-14 18:36:02
 
 
import tensorflow as tf
import numpy as np
import os

train_folder = '/home/jiang/AI_Practice/Mnist/my_mnist/cifar-10/train'
test_folder = '~/AI_Practice/Mnist/my_mnist/cifar-10/test'



def ReadFile(path):
    pic_folder = os.listdir(path)
    for folder in pic_folder:
        pic_folder_path = path + "/" + folder
        files = os.listdir(pic_folder_path)
        train_dataset_label = open((path + '.txt'),'w')
    print pic_folder

def WriteFile():
    print "helloworld"


ReadFile(train_folder)


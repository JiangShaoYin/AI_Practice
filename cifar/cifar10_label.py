#        coding:utf-8
#	 @file    mnist_label.py
#	 @author  1801210547_江绍印(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-14 18:36:02
 
import tensorflow as tf
import numpy as np
import os

train_folder = '/home/jiang/AI_Practice/cifar/cifar-10/train'
test_folder = '/home/jiang/AI_Practice/cifar/cifar-10/test'

def ReadFile(path):
    pic_folder = os.listdir(path) #pic_folder is a list which equals to  [bird, dog, ....]

    dict_pic_num ={'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
                   'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

    train_dataset_label = open((path + '_label.txt'),'w')

    for folder in pic_folder: # folder like fog ,cat ,etc 
        pic_folder_path = path + "/" + folder
        #open folder in ./train, such as frog cat bird etc
        files = os.listdir(pic_folder_path)
        
        for filename in files:#filename == batch_2_num_6379.jpg
            train_dataset_label.write(filename + '_' + folder + ' ' + str(dict_pic_num[folder]) +'\n')

    print pic_folder

def WriteFile():
   print "helloworld"


ReadFile(train_folder)
ReadFile(test_folder)


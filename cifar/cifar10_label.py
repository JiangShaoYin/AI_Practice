#        coding:utf-8
#	 @file    mnist_label.py
#	 @author  1801210547_江绍印(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-14 18:36:02
 
import tensorflow as tf
import numpy as np
import os
#script
train_folder = '/home/jiang/AI_Practice/cifar/cifar-10/train'
test_folder = '/home/jiang/AI_Practice/cifar/cifar-10/test'

def ReadFile(path):
    pic_folder = os.listdir(path) #pic_folder is a list which equals to  [bird, dog, ....]
    num_dict = [0,1,2,3,4,5,6,7,8,9]
    dict_pic_num = dict(zip(pic_folder,num_dict))#combine two list into one dict
    train_dataset_label = open((path + '_label.txt'),'w')
    for folder in pic_folder: # folder like fog ,cat ,etc 
        #pic_folder_path = path + "/" + folder
        pic_folder_path = os.path.join(path, folder)
        #open folder in ./train, such as frog cat bird etc
        files = os.listdir(pic_folder_path)
        for filename in files:#filename == batch_2_num_6379.jpg
            train_dataset_label.write(filename + ' ' + folder + ' ' + str(dict_pic_num[folder]) +'\n')
    print 'write successfully!!!'

ReadFile(train_folder)
ReadFile(test_folder)


#        coding:utf-8
#	 @file    mnist_label.py
#	 @author  1801210547_江绍印(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-14 18:36:02
 
import tensorflow as tf
import numpy as np
import os
#script
#train_folder = '/home/jiang/AI_Practice/cifar/cifar-10/train'
#test_folder = '/home/jiang/AI_Practice/cifar/cifar-10/test'
train_folder = './cifar-10/train'
test_folder = './cifar-10/test'

def ReadFile(path):
    pic_folder = os.listdir(path)                   #s.listdir返回文件夹内的文件列表，[bird, dog, ....]
    num_dict = [0,1,2,3,4,5,6,7,8,9]
    dict_pic_num = dict(zip(pic_folder,num_dict)  ) #把2个列表合成一个字典
    train_dataset_label = open((path + '_label.txt'),'w')
    for folder in pic_folder:                       # folder分别是fog ,cat等文件夹 
        pic_folder_path = os.path.join(path, folder)#文件夹路径拼接
        files = os.listdir(pic_folder_path)         #files是动物文件（dog,cat.....）下图片集合的列表
        for filename in files:                      #filename == batch_2_num_6379.jpg
            train_dataset_label.write(filename + ' ' + folder + ' ' + str(dict_pic_num[folder]) +'\n')
    print 'write successfully!!!'

ReadFile(train_folder)
ReadFile(test_folder)


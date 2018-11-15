 
#        coding:utf-8
#	 @file    mnist_generate.py
#	 @author  1801210547_江绍印(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-12 14:51:28
 
 
import tensorflow as tf
import numpy as np
import os
from PIL import Image

image_train_path = './cifar-10/cifar_train_jpg_60000/'
label_train_path = './cifar-10/mnist_train_jpg_60000.txt'
tfRecord_train = './data/cifar_train.tfrecords'

image_test_path = './cifar-10/cifar_test_jpg_10000/'
label_test_path = './cifar-10/cifar_test_jpg_10000.txt'
tfRecord_test = './data/cifar_test.tfrecords'

data_path = './data'
resize_height = 28
resize_width = 28


def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    animal_folders = f.readlines()
    
    f.close()
    for animal_folder in animal_folders:
        f1 = open((label_path+'/'+animal_folder), 'r')
        contents = f1.readlines()
        f1.close()
        for content in contents:
            value = content.split()
            img_path = image_path + value[0]
            img = Image.open(img_path)
            img_raw = img.tobytes()
            labels = [0] * 10
            labels[int(value[1])] = 1

            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                }))
            writer.write(example.SerializeToString())
            num_pic += 1
            print ("the num of pic :",num_pic)
        writer.close()
    print("write tfRecord sussfully")
    
write_tfRecord(tfRecord_train, image_train_path, label_train_path)

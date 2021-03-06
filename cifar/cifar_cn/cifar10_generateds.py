 
#        coding:utf-8
#	 @file    mnist_generate.py
#	 @author  1801210547_江绍印(jiangshaoyin@pku.edu.cn)
#	 @date    2018-11-12 14:51:28
 
 
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random

image_train_path = './cifar-10/train/'						# 训练集路径
label_train_path = './cifar-10/train_label.txt'		# 存放测试集中图片相关信息的文件路径 
tfRecord_train = './data/cifar_train.tfrecords'		# 训练集的tfrecord文件路径

image_test_path = './cifar-10/test/'							# 测试集路径
label_test_path = './cifar-10/test_label.txt'  		# 存放训练集中图片相关信息的文件路径
tfRecord_test = './data/cifar_test.tfrecords'			# 测试集的tfrecord文件路径

data_path = './data'
resize_height = 32
resize_width = 32


def write_tfRecord(tfRecordName, image_path, label_path):	# 生成tfrecord文件，对图片进行标注，将图和标签封装进example并做序列化处理，写入tfrecord文件
    writer = tf.python_io.TFRecordWriter(tfRecordName)		#新建一writer
    num_pic = 0
    f = open(label_path, 'r')
    contents =f.readlines()
    random.shuffle(contents)
    f.close()
    for content in contents:															#遍历每张图并进行标注，将图和标签封装进example中
        value = content.split()

        img_path = image_path + '/' + value[1] + '/' + value[0]

        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[2])] = 1

        example = tf.train.Example(features=tf.train.Features(feature={        								 #tf.train.Example用来存储训练数据,训练数据的特征用键值对的形式表示
                'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                }))
        writer.write(example.SerializeToString())																								#将example序列化，并通过writer写入文件
        num_pic += 1
        print ("the num of pic :",num_pic)
    writer.close()
    print("write tfRecord sussfully")

def generate_tfRecord():																										#生成训练集和测试集的tfrecord文件
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print 'the directory was created sussfully'
    else:
        print 'directory already exists'
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)

def read_tfRecord(tfRecord_path):																						#读取并解析tfrecord文件
    filename_queue = tf.train.string_input_producer([tfRecord_path])				#该函数会生成一个先入先出的队列，文件阅读器会使用它来读取数据
    reader = tf.TFRecordReader()																						#新建一个reader
    _, serialized_example = reader.read(filename_queue)       						  #把读出的每个样本保存在 serialized_example 中进行解序列化，标签和图片的
                                                                						#键名应该和制作 tfrecords 的键名相同，其中标签给出几分类
    features = tf.parse_single_example(serialized_example,									#将 tf.train.Example 协议内存块(protocol buffer)解析为张量
                                        features={
                                        'label':tf.FixedLenFeature([10], tf.int64),
                                        'img_raw':tf.FixedLenFeature([], tf.string)
                                        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)											#将 img_raw 字符串转换为 8 位无符号整型
    img.set_shape([3072])																										#将形状变为 1 行 3072 列
    img =tf.cast(img, tf.float32)* (1./255)																	#把像素值变成 0 到 1 之间的浮点数
    label = tf.cast(features['label'], tf.float32)													#把标签列表变为浮点数
    print 'read successfully!'
    return img, label   																										#返回图和标签

def get_tfrecord(num, isTrain =True):																				#随机读取一个batch的训练数据或一个batch的测试数据
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    
    img_batch, label_batch = tf.train.shuffle_batch([img, label],						#随机读取一个batch的数据
                                                    batch_size = num,
                                                    num_threads = 2,
                                                    capacity = 1000,
                                                    min_after_dequeue = 700)
    print 'geting successfully!'
    return img_batch,label_batch

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()

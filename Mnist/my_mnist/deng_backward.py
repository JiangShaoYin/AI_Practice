#coding:utf-8
#传入各个模块
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
import deng_generateds
#每一组喂入数据大小
BATCH_SIZE = 200
#学习率初始值
LEARNING_RATE_BASE = 0.1
#学习率衰减率
LEARNING_RATE_DECAY = 0.99
#正则化参数值
REGULARIZER = 0.0001
#总训练轮数
STEPS = 50000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
#训练模型存储路径
MODEL_SAVE_PATH = "./model/"
#训练模型文件名
MODEL_NAME = "mnist_model"
#定义反向传播函数
train_num_examples = 60000
def backward():
	#定义输入
	x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
	#定义输出
	y_ = tf.placeholder(tf.float32, [None,mnist_forward.OUTPUT_NODE])
	#将x和正则化参数传入前向神经网络，得到y值
	y = mnist_forward.forward(x, REGULARIZER)
	#初始化轮数
	global_step = tf.Variable(0, trainable=False)
	#定义并得出交叉熵的值
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cem = tf.reduce_mean(ce)
	#定义损失函数
	loss = cem + tf.add_n(tf.get_collection('losses'))
	#定义指数下降学习率
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		train_num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)
	#定义反向传播方法为梯度下降法
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	#定义滑动平均值
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	#对所有待训练参数求滑动平均
	ema_op = ema.apply(tf.trainable_variables())
	#实现滑动平均与训练过程同步运行
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name='train')
	#设置断点
	saver = tf.train.Saver()
	img_batch, label_batch = deng_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)#3

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		#实现断点续训
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		coord = tf.train.Coordinator()#4
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)#5

		for i in range(STEPS):
			xs, ys = sess.run([img_batch, label_batch])
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
			#每1000轮输出损失函数值同时保存断点
			if i % 1000 == 0:
                                print "........................."
				print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
				#将断点保存到相应路径
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
		
		coord.request_stop()#7
		coord.join(threads)#8
def main():
	#读取数据集
	backward()

if __name__ == '__main__':
	main()

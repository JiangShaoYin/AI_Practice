#coding:utf-8
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import cifar10_lenet5_generateds
BATCH_SIZE = 100

save_dir = r"./train_image_63.model"
batch_size_ = 100
lr = tf.Variable(0.0001, dtype=tf.float32)
x = tf.placeholder(tf.float32, [None, 32,32,3])
y_ = tf.placeholder(tf.float32, [None]) #

img_batch, label_batch = cifar10_lenet5_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)#img_batch = [[32,32,3],[32,32,3]], label_batch=[[0,0,0,1,0...], [0,1,0....]]
#with tf.Session() as sess:#???
#    print sess.run(img_batch), sess.run(label_batch)#???

# 将label值进行onehot编码,10分类
one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=10)
#one_hot_labels = tf.argmax(y_,1)
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=10, is_training=True)
pred = tf.reshape(pred, shape=[-1, 10]) #预测值，相当于y
# 定义损失函数和优化器
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=one_hot_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# 准确度
a = tf.argmax(pred, 1)
b = tf.argmax(one_hot_labels, 1)
correct_pred = tf.equal(a, b)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()                                 # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 启动QueueRunner,此时文件名队列已经进队  
    i = 0
    while True:
        i += 1
        b_image, b_label = sess.run([img_batch, label_batch])
        _, loss_, y_t, y_p, a_, b_ = sess.run([optimizer, loss, one_hot_labels, pred, a, b], feed_dict={x: b_image,
                                                     y_: b_label})
        #print('step: {}, train_loss: {}'.format(i, loss_))
        if i % 100 == 0:
            _loss, acc_train = sess.run([loss, accuracy], feed_dict={x: b_image, y_: b_label})
            print('--------------------------------------------------------')
            print('step: {}  train_acc: {}  loss: {}'.format(i, acc_train, _loss))
            print('--------------------------------------------------------')
          #  if i == 200000:
            saver.save(sess, save_dir, global_step=i)
           # elif i == 300000:
        #    saver.save(sess, save_dir, global_step=i)
            #elif i == 400000:
             #   saver.save(sess, save_dir, global_step=i)
             #   break
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)

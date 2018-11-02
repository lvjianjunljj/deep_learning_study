#!/usr/bin/python
# coding:utf8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

emotion ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

data = pd.read_csv(r'D:\\DeepLearning\\expressionRecognition\\jaffe\\face.csv', dtype='a')
label = np.array(data['emotion'])
img_data = np.array(data['pixels'])
N_sample = label.size
Face_data = np.zeros((N_sample, 48*48))
Face_label = np.zeros((N_sample, 7), dtype=int)


for i in range(N_sample):
    x = img_data[i]
    x = np.fromstring(x, dtype=float, sep=' ')
    x = x/x.max()
    Face_data[i] = x
    Face_label[i, int(label[i])] = 1

# 参数
dropout = 0.5
class_sum = 7

# dropout减轻过拟合问题
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 48*48])
y = tf.placeholder(tf.float32, [None, class_sum])


def conv_pool_layer(data, weights_size, biases_size):
    weights = tf.Variable(tf.truncated_normal(weights_size, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=biases_size))
    conv2d = tf.nn.conv2d(data, weights, strides=[1,1,1,1], padding='SAME')
    relu = tf.nn.relu(conv2d + biases)
    return tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def linear_layer(data, weights_size, biases_size):
    weights = tf.Variable(tf.truncated_normal(weights_size, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=biases_size))
    return tf.add(tf.matmul(data, weights), biases)


def convolutional_neural_network(x, keep_prob):
    with tf.name_scope('input'):
        x_image=tf.reshape(x, [-1,48,48,1])
    with tf.name_scope('conv1'):
        h_pool1=conv_pool_layer(x_image, [5,5,1,32], [32])
    with tf.name_scope('conv2'):
        h_pool2=conv_pool_layer(h_pool1, [5,5,32,64], [64])
    h_pool2_flat=tf.reshape(h_pool2, [-1, 12*12*64])
    with tf.name_scope('fc3'):
        h_fc1=tf.nn.relu(linear_layer(h_pool2_flat, [12*12*64,1024], [1024]))
    with tf.name_scope('dropout4'):
        h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope('softmax5'):
        out = tf.nn.softmax(linear_layer(h_fc1_drop, [1024,class_sum], [class_sum]))
    return out

pred = convolutional_neural_network(x, keep_prob)

# #======取前200个作为训练数据==================
train_num = 200
test_num = 13

train_x = Face_data [0:train_num, :]
train_y = Face_label [0:train_num, :]

test_x =Face_data [train_num : train_num+test_num, :]
test_y = Face_label [train_num : train_num+test_num, :]

batch_size = 20
train_batch_num = train_num / batch_size
test_batch_num = test_num / batch_size

def batch_data(x, y, batch, num):
    ind = np.arange(num)
    index = ind[batch * batch_size:(batch + 1) * batch_size]
    batch_x = x[index, :]
    batch_y = y[index, :]
    return batch_x,batch_y

# 训练和评估模型
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y*tf.log(pred))
    tf.summary.histogram("cross_entropy", cross_entropy)
    # tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope('minimize'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_pred'):
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
        tf.summary.histogram("accuracy", accuracy)
    # 输出包含单个标量值的摘要协议缓冲区
    tf.summary.scalar('accuracy', accuracy)


total_train_loss = []
total_train_acc = []
total_test_loss = []
total_test_acc = []

train_epoch = 50
# 合并在默认图形中收集的所有摘要
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("D:\\DeepLearning\\expressionRecognition\\jaffe\\logs", sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(0, train_epoch):
        Total_train_loss = 0
        Total_train_acc = 0
        for train_batch in range (0, int(train_batch_num)):
            batch_x,batch_y = batch_data(train_x, train_y, train_batch, train_num)
            # 优化操作
            # sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            summary, _ = sess.run([merged,train_step], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            writer.add_summary(summary,train_batch)

            if train_batch % batch_size == 0:
                # 计算损失和准确率
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

                print("Epoch: " + str(epoch+1) + ", Batch: "+ str(train_batch) +
                      ", Loss= " + "{:.3f}".format(loss) +
                      ", Training Accuracy= " + "{:.3f}".format(acc))
                Total_train_loss = Total_train_loss + loss
                Total_train_acc = Total_train_acc + acc

        total_train_loss.append(Total_train_loss)
        total_train_acc.append(Total_train_acc)

    writer.close()


plt.subplot(2,1,1)
plt.ylabel('Train loss')
plt.plot(total_train_loss, 'r')
plt.subplot(2,1,2)
plt.ylabel('Train accuracy')
plt.plot(total_train_acc, 'r')
plt.savefig("face_loss_acc.png")
plt.show()

# use command line to input
# tensorboard --logdir='D:\\DeepLearning\\expressionRecognition\\jaffe\\logs'
# open the tensorboard
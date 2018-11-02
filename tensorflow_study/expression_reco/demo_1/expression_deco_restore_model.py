#!/usr/bin/python
# coding:utf8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

data = pd.read_csv(r'D:\\DeepLearning\\expressionRecognition\\jaffe\\face.csv', dtype='a')
label = np.array(data['emotion'])
img_data = np.array(data['pixels'])
N_sample = label.size
Face_data = np.zeros((N_sample, 48 * 48))
Face_label = np.zeros((N_sample, 7), dtype=int)

for i in range(N_sample):
    x = img_data[i]
    x = np.fromstring(x, dtype=float, sep=' ')
    x = x / x.max()
    Face_data[i] = x
    Face_label[i, int(label[i])] = 1

# 参数
dropout = 0.5
class_sum = 7

# dropout减轻过拟合问题
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 48 * 48])
y = tf.placeholder(tf.float32, [None, class_sum])


def conv_pool_layer(data, weights_size, biases_size):
    weights = tf.Variable(tf.truncated_normal(weights_size, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=biases_size))
    conv2d = tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv2d + biases)
    return tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def linear_layer(data, weights_size, biases_size):
    weights = tf.Variable(tf.truncated_normal(weights_size, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=biases_size))
    return tf.add(tf.matmul(data, weights), biases)


def convolutional_neural_network(x, keep_prob):
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    h_pool1 = conv_pool_layer(x_image, [5, 5, 1, 32], [32])
    h_pool2 = conv_pool_layer(h_pool1, [5, 5, 32, 64], [64])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
    h_fc1 = tf.nn.relu(linear_layer(h_pool2_flat, [12 * 12 * 64, 1024], [1024]))

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return tf.nn.softmax(linear_layer(h_fc1_drop, [1024, class_sum], [class_sum]))


pred = convolutional_neural_network(x, keep_prob)

# #======取前200个作为训练数据==================
train_num = 200
test_num = 13

train_x = Face_data[0:train_num, :]
train_y = Face_label[0:train_num, :]

test_x = Face_data[train_num: train_num + test_num, :]
test_y = Face_label[train_num: train_num + test_num, :]

batch_size = 20
train_batch_num = train_num / batch_size
test_batch_num = test_num / batch_size


def batch_data(x, y, batch, num):
    ind = np.arange(num)
    index = ind[batch * batch_size:(batch + 1) * batch_size]
    batch_x = x[index, :]
    batch_y = y[index, :]
    return batch_x, batch_y


# 训练和评估模型
cross_entropy = -tf.reduce_sum(y * tf.log(pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

total_train_loss = []
total_train_acc = []
total_test_loss = []
total_test_acc = []

train_epoch = 50
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, tf.train.latest_checkpoint('D:/DeepLearning/expressionRecognition/modelGallery/jaffe/'))
    batch_x, batch_y = batch_data(train_x, train_y, 0, train_num)
    loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    print("Loss= " + "{:.3f}".format(loss) +
          ", Training Accuracy= " + "{:.3f}".format(acc))

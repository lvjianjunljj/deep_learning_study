#!/usr/bin/env python  
# _*_ coding: utf-8 _*_  

import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data


# 定义神经网络模型的评估部分
def compute_accuracy(test_xs, test_ys):
    # 使用全局变量prediction  
    global prediction
    # 获得预测值y_pre  
    y_pre = sess.run(prediction, feed_dict={xs: test_xs, keep_prob: 1})
    # 判断预测值y和真实值y_中最大数的索引是否一致，y_pre的值为1-10概率, 返回值为bool序列  
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(test_ys, 1))
    # 定义准确率的计算  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast将bool转换为float32
    # 计算准确率  
    result = sess.run(accuracy)
    return result


# 下载mnist数据
mnist = input_data.read_data_sets('..\MNIST_data', one_hot=True)


# 权重参数初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断的正态分布，标准差stddev
    return tf.Variable(initial)


# 偏置参数初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层
def conv2d(x, W):
    # input的四个参数：[batch, height, width, channels], [batch_size, image_rows, image_cols, number_of_colors]
    # height, width就是图像的高度和宽度，batch和channels在卷积层中通常设为1  
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 输入输出数据的placeholder
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
# dropout的比例  
keep_prob = tf.placeholder(tf.float32)

# 对数据进行重新排列，形成图像  
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1, 28, 28, 1

print(x_image.shape)

# 卷积层一  
# patch为5*5，in_size为1，即图像的厚度，如果是彩色，则为3，32是out_size，输出的大小-》32个卷积和（滤波器）  
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# ReLU操作，输出大小为28*28*32  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Pooling操作，输出大小为14*14*32  
h_pool1 = max_pool_2x2(h_conv1)

# 卷积层二  
# patch为5*5，in_size为32，即图像的厚度，64是out_size，输出的大小  
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# ReLU操作，输出大小为14*14*64  
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# Pooling操作，输出大小为7*7*64  
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层一  
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 输入数据变换  
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 整形成m*n,列n为7*7*64
# 进行全连接操作  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # tf.matmul  
# 防止过拟合，dropout  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层二
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 预测  
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 计算loss  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# 神经网络训练
# GradientDescentOptimizer
# AdagradOptimizer
# AdagradDAOptimizer
# MomentumOptimizer
# AdamOptimizer
# FtrlOptimizer
# RMSPropOptimizer
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 定义Session  
sess = tf.Session()
init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=1)
tf.add_to_collection('x', xs)
tf.add_to_collection('prediction', prediction)
tf.add_to_collection('keep_prob', keep_prob)

# 执行初始化
sess.run(init)

# 进行训练迭代
for i in range(1000):
    # 取出mnist数据集中的100个数据
    batch_xs, batch_ys = mnist.train.next_batch(50)  # 100
    # 执行训练过程并传入真实数据
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 100 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
saver.save(sess, "D:/DeepLearning/digitalRecognition/modelGallery/defineGraphTF/MyModel.ckpt", global_step=1000)
sess.close()

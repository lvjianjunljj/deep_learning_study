#!/usr/bin/python
# coding:utf8

import cv2
import os
import numpy as np
import tensorflow as tf


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


# Stump-based 20x20 gentle adaboost frontal face detector.
cascade = cv2.CascadeClassifier(".\\haarcascade_frontalface_alt.xml")
print(cascade.empty())

f = "D:\\DeepLearning\\expressionRecognition\\jaffe\\jaffe\\"
fs = os.listdir(f)
data = np.zeros([213, 48 * 48], dtype=np.uint8)
label = np.zeros([213], dtype=int)
i = 0
for f1 in fs:
    tmp_path = os.path.join(f, f1)
    if not os.path.isdir(tmp_path):
        # print(tmp_path[len(f):])
        img = cv2.imread(tmp_path, 1)
        dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detect(dst, cascade)
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1 + 10, y1 + 20), (x2 - 10, y2), (0, 255, 255), 2)
            # 调整截取脸部区域大小
            # img_roi = np.uint8([y2 - (y1 + 20), (x2 - 10) - (x1 + 10)])
            roi = dst[y1 + 20:y2, x1 + 10:x2 - 10]
            # img_roi = roi
            re_roi = cv2.resize(roi, (48, 48))
            # 获得表情label
            img_label = tmp_path[len(f) + 3:len(f) + 5]
            # print(img_label)
            if img_label == 'AN':
                label[i] = 0
            elif img_label == 'DI':
                label[i] = 1
            elif img_label == 'FE':
                label[i] = 2
            elif img_label == 'HA':
                label[i] = 3
            elif img_label == 'SA':
                label[i] = 4
            elif img_label == 'SU':
                label[i] = 5
            elif img_label == 'NE':
                label[i] = 6
            else:
                print("get label error.......\n")

            data[i][0:48 * 48] = np.ndarray.flatten(re_roi)
            i = i + 1


emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# 参数
dropout = 0.5
class_sum = 7

# dropout减轻过拟合问题
x = tf.placeholder(tf.float32, [48 * 48])
y = tf.placeholder(tf.float32)


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


pred = convolutional_neural_network(x, 1)

saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, tf.train.latest_checkpoint('D:/DeepLearning/expressionRecognition/modelGallery/jaffe/'))
    prediction = tf.argmax(pred, 1)
    count = 0
    for j in range(len(label)):
        predint = prediction.eval(feed_dict={x: data[j], y: 1.0}, session=sess)
        print(str(label[j]) + ' recognize result: ' + str(predint[0]))
        if str(label[j]) != str(predint[0]):
            count += 1

print(count)

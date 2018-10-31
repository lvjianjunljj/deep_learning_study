#!/usr/bin/env python
# 导入mnist数据库
import tensorflow as tf
import cv2
import numpy as np

# 创建会话
sess = tf.Session()
new_saver = tf.train.import_meta_graph(
    r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2/MyModel.ckpt.meta')
new_saver.restore(sess, r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2/MyModel.ckpt')
x = tf.get_collection('x')[0]
y = tf.get_collection('y')[0]
print("model restore success!!!")
# 取出一个测试图片
for i in range(20):
    image_path = r"D:/DeepLearning/digitalRecognition/digitalData/test/" + str(i) + ".bmp"
    img = cv2.imread(image_path, 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    flatten_img = np.reshape(shrink, 784)

    # 根据模型计算结果
    ret = sess.run(y, feed_dict={x: flatten_img.reshape(1, 784)})
    print("计算模型结果成功！")
    # 显示测试结果
    print("predict result: %d" % (ret.argmax()))
    print("actual result: %d" % (i % 10))

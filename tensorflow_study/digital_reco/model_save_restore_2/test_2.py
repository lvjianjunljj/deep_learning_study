# The most successful model restore demo!!!

import tensorflow as tf
import cv2
import numpy as np
import random

# create session
sess = tf.Session()
new_saver = tf.train.import_meta_graph(
    r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2/MyModel.ckpt.meta')
new_saver.restore(sess, r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2/MyModel.ckpt')
# Automatically get the last saved model
# Use this function when you save the model every few steps
# new_saver.restore(sess, tf.train.latest_checkpoint(r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2'))
x = tf.get_collection('x')[0]
y = tf.get_collection('y')[0]
print("model restore success!!!")
for i in range(20):
    # get a test image
    image_path = r"D:/DeepLearning/digitalRecognition/digitalData/test/" + str(i) + ".bmp"
    img = cv2.imread(image_path, 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    flatten_img = np.reshape(shrink, 784)

    # calculate results based on the model
    ret = sess.run(y, feed_dict={x: flatten_img.reshape(1, 784)})
    # show the test result
    print("actual result: %d" % (i % 10), end='\t')
    print("predict result: %d" % (ret.argmax()))

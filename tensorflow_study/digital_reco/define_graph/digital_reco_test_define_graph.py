import tensorflow as tf

import numpy as np
import cv2

# create session
sess = tf.Session()
new_saver = tf.train.import_meta_graph(
    'D:/DeepLearning/digitalRecognition/modelGallery/defineGraphTF/MyModel.ckpt-1000.meta')
# new_saver.restore(sess, 'D:/DeepLearning/digitalRecognition/modelGallery/defineGraphTF/MyModel.ckpt-1000')
# Automatically get the last saved model
# Use this function when you save the model every few steps
new_saver.restore(sess, tf.train.latest_checkpoint(r'D:/DeepLearning/digitalRecognition/modelGallery/defineGraphTF/'))
x = tf.get_collection('x')[0]
keep_prob = tf.get_collection('keep_prob')[0]
prediction = tf.get_collection('prediction')[0]

print("model restore success!!!")
for i in range(20):
    # get a test image
    image_path = r"D:/DeepLearning/digitalRecognition/digitalData/test/" + str(i) + ".bmp"
    img = cv2.imread(image_path, 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    flatten_img = np.reshape(shrink, 784)

    # calculate results based on the model
    ret = sess.run(prediction, feed_dict={x: flatten_img.reshape(1, 784), keep_prob: 1})
    # show the test result
    print("actual result: %d" % (i % 10), end='\t')
    print("predict result: %d" % (ret.argmax()))

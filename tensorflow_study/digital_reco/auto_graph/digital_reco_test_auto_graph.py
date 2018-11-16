from PIL import Image, ImageFilter

import tensorflow as tf

import matplotlib.pyplot as plt

import cv2


def negate784(data):
    for x in range(784):
        data[x] = 1 - data[x]


def showData2828(data):
    for x in range(28):
        for y in range(28):
            if data[x * 28 + y] < 0.1:
                print('.', end='\t')
            else:
                print(1, end='\t')
        print()


def imageprepare(index):
    img = cv2.imread("D:/DeepLearning/digitalRecognition/digitalData/test/" + str(index) + ".bmp", 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # show the image with matplotlib
    # plt.imshow(shrink)
    # plt.show()
    return shrink


# Define the model (same as when creating the model file)


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(
        r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2/MyModel.ckpt.meta')
    new_saver.restore(sess, r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2/MyModel.ckpt')
    # Use this function when you save the model every few steps
    # new_saver.restore(sess,
    #                   tf.train.latest_checkpoint(r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2'))
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    for i in range(20):
        flatten_img = imageprepare(i)
        ret = sess.run(y, feed_dict={x: flatten_img.reshape(1, 784)})
        # show the test result
        print("actual result: %d" % (i % 10), end='\t')
        print("predict result: %d" % (ret.argmax()))

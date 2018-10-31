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
            # print(data[x * 28 + y], end='\t')
            if data[x * 28 + y] < 0.1:
                print('.', end='\t')
            else:
                print(1, end='\t')
        print()


def imageprepare(index):
    img = cv2.imread("D:/DeepLearning/digitalRecognition/digitalData/" + str(index) + ".bmp", 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite("D:/DeepLearning/digitalRecognition/digitalData/test.jpg", shrink)

    file_name = 'D:/DeepLearning/digitalRecognition/digitalData/test.jpg'  # 导入自己的图片地址
    file_name = "D:/DeepLearning/digitalRecognition/digitalData/train/train_" + str(index) + ".bmp"
    im = Image.open(file_name).convert('L')
    im.save("D:/DeepLearning/digitalRecognition/digitalData/simple.JPG")

    # plt.imshow(im)
    # plt.show()
    tv = list(im.getdata())  # get pixel values
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [x * 1.0 / 255.0 for x in tv]
    # print(tva)
    return tva


"""

This function returns the predicted integer.

The input is the pixel values from the imageprepare() function.

"""

# Define the model (same as when creating the model file)

x = tf.placeholder(tf.float32, [None, 784])
# paras
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32)
# init
init = tf.initialize_all_variables()

saver = tf.train.Saver(max_to_keep=1)
# sess = tf.Session()
# sess.run(init_op)
# saver.restore(sess, "D:/test/MyModel.ckpt-1000.data-00000-of-00001")#这里使用了之前保存的模型参数

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('D:/DeepLearning/digitalRecognition/modelGallery/autoGraphTF/'))
    for i in range(20):
        result = imageprepare(i)
        # showData2828(result)
        # negate784(result)
        sess.run(init)
        # saver = tf.train.import_meta_graph('D:/DeepLearning/digitalRecognition/modelGallery/autoGraphTF/MyModel.ckpt-1000.meta')
        prediction = tf.argmax(y, 1)
        predint = prediction.eval(feed_dict={x: [result], y_: 1.0}, session=sess)

        print(str(i) + ' recognize result: ' + str(predint[0]))

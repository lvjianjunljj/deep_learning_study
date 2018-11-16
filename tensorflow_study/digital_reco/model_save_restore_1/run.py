import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow_study.digital_reco.model_save_restore_1.model import Network
import cv2


class Predict:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.restore()  # 加载模型到sess中

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(r'D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore1')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("未保存任何模型")

    def predict_test(self, image_path):
        # 读图片并转为黑白的
        # img = Image.open(image_path).convert('L')
        img = cv2.imread(image_path, 0)
        shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        flatten_img = np.reshape(shrink, 784)
        # for i in range(784):
        #     flatten_img[i] = 0 if flatten_img[i] > 80 else (255 - flatten_img[i])
        # for i in range(28):
        #     for j in range(28):
        #         print(flatten_img[i * 28 + j], end='\t')
        #     print()

        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得独热编码最大值的下标，即代表的数字
        print(image_path)
        print('        -> Predict digit', np.argmax(y[0]))

    def predict(self, image_path):
        # 读图片并转为黑白的
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, 784)
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得独热编码最大值的下标，即代表的数字
        print(image_path)
        print('        -> Predict digit', np.argmax(y[0]))


if __name__ == "__main__":
    app = Predict()
    for i in range(20):
        image_path = r"D:/DeepLearning/digitalRecognition/digitalData/test/" + str(i) + ".bmp"
        # image_path = r"D:\DeepLearning\digitalRecognition\digitalData\train\train_" + str(i) + ".bmp"
        # image_path = r"D:\DeepLearning\digitalRecognition\digitalData\test.jpg"
        app.predict_test(image_path)

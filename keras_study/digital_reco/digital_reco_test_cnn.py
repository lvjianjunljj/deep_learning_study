from keras.models import load_model
import numpy as np
from PIL import Image
import cv2


def showData2828(data):
    for x in range(28):
        for y in range(28):
            if data[0][x][y][0] > 0.85:
                print('.', end='\t')
            else:
                print(1, end='\t')
        print()


def negate2828(data):
    for x in range(28):
        for y in range(28):
            data[0][0][x][y] = 1 - data[0][0][x][y]


model = load_model('D:/DeepLearning/digitalRecognition/modelGallery/keras/my_model_CNN2.h5')
for i in range(20):
    numName = str(i)
    img = cv2.imread("D:/DeepLearning/digitalRecognition/digitalData/" + numName + ".bmp", 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # data = np.expand_dims(shrink, axis=2)
    # data = np.array(data, dtype=np.float32) / 255.0
    # data = np.expand_dims(data, axis=0)
    # data = np.array(shrink, dtype=np.float32) / 255
    # data = data.reshape(1, 1, 28, 28)
    data = shrink.reshape(1, 1, 28, 28)
    negate2828(data)
    print(str(i) + ' recognize result: ' + str(model.predict_classes(data, batch_size=1, verbose=0)[0]))

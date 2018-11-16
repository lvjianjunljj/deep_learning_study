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


def show2828(data):
    for x in range(28):
        for y in range(28):
            print(data[0][x][y][0], end='\t')
        print()


def negate2828(data):
    for x in range(28):
        for y in range(28):
            data[0][x][y][0] = 1 - data[0][x][y][0]


def negate784(data):
    for x in range(784):
        data[0][x] = 1 - data[0][x]


model = load_model('D:/DeepLearning/digitalRecognition/modelGallery/keras/my_model.h5')
for i in range(20):
    numName = str(i)
    img = cv2.imread(r"D:/DeepLearning/digitalRecognition/digitalData/test/" + numName + ".bmp", 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # data = np.expand_dims(shrink, axis=2)
    # data = np.array(data, dtype=np.float32) / 255.0
    # data = np.expand_dims(data, axis=0)
    # data = np.array(shrink, dtype=np.float32) / 255
    # data = data.reshape(1, 784)
    # data = data.reshape(1, 28, 28, 1)

    # Expand_dims(a, axis) is to add the data on the axis of the axis,
    # this data is in the 0 position of the axis of axis
    # For example, there are two data that were originally one-dimensional,
    # if axis=0, shape will be (1,2), if axis=1, shape will be (2,1)
    # For example if shape was(2,3),axis=0,shape will be (1,2,3),if axis=1, then shape will be (2,1,3)
    # So I think function Expand_dims() is unnecessary here.

    data = shrink.reshape(1, 28, 28, 1)
    # Function predict_classes predicts the category, and the value is the category.
    # It only can be used for sequence model prediction, not for functional models
    # print(str(i) + ' recognize result: ' + str(model.predict_classes(data, batch_size=1, verbose=0)[0]))

    # Function predict predicts the value, and the output is still 10 code values(output was encoded with one hotF).
    # After the prediction, the index number can be obtained by the function numpy.argmax().
    predict = model.predict(data, batch_size=1, verbose=0)[0]
    print(str(i) + ' recognize result: ' + str(np.argmax(predict)))

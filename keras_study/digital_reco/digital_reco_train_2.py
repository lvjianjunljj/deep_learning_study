from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
from keras.datasets import mnist
import numpy as np

model = Sequential()
model.add(Dense(500, input_shape=(784, )))  # 输入层， 28*28=784
model.add(Activation('tanh'))
model.add(Dropout(0.5))  # 50% dropout

model.add(Dense(500))  # 隐藏层， 500
model.add(Activation('tanh'))
model.add(Dropout(0.5))  # 50% dropout

model.add(Dense(10))  # 输出结果， 10
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 设定学习效率等参数
model.compile(loss='categorical_crossentropy', optimizer=sgd)  # 使用交叉熵作为loss
# model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')  # 使用交叉熵作为loss

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 使用mnist读取数据（第一次需要下载）
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# y_test = keras_study.utils.to_categorical(y_test, 10)
# y_train = keras_study.utils.to_categorical(y_train, 10)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

y_train = (np.arange(10) == y_train[:, None]).astype(int)  # 将index转换橙一个one_hot矩阵
y_test = (np.arange(10) == y_test[:, None]).astype(int)

model.fit(x_train, y_train, batch_size=200, epochs=10, shuffle=True, verbose=1, validation_split=0.3)

print("test set")
scores = model.evaluate(x_test, y_test, batch_size=200, verbose=1)
print("")
print("The test loss is %f" % scores)
result = model.predict(x_test, batch_size=200, verbose=1)

result_max = np.argmax(result, axis=1)
test_max = np.argmax(y_test, axis=1)

result_bool = np.equal(result_max, test_max)
true_num = np.sum(result_bool)
print("")
print("The accuracy of the model is %f" % (true_num / len(result_bool)))
model.save('D:/DeepLearning/digitalRecognition/modelGallery/keras/my_model2.h5')

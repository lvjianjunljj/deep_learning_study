from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical

# 使用keras内置的函数加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 对数据进行resize来满足网络的输入
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train / 255.0
X_test = X_test / 255.0
# 对标签进行one-hot编码
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 使用序贯模型
model = Sequential()
# 定义网络结构
model.add(Conv2D(input_shape=(28, 28, 1), filters=6, kernel_size=(5, 5), padding="valid", activation="tanh"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # 比较关键的一层，将卷积结果“压平”，即将多维结果一维化，好过渡到全连接层
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))
# 优化方法
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
# 编译：使用的SGD优化方法，多类的对数损失，指标是accuracy
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# 训练
history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)
# 保存网络结构和权值
model.save('D:/DeepLearning/digitalRecognition/modelGallery/keras/my_model3.h5')

import torch
import torch.nn as nn
import cv2
import io
import numpy as np


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
            data[0][x][y][0] = 1 - data[0][x][y][0]


def negate784(data):
    for x in range(784):
        data[0][x] = 1 - data[0][x]


# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 guaranteed the sizes of input and output are the same
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # Define the forward propagation process, input is x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # The input and output of nn.Linear() are all values of dimension one,
        # so we should flatten the multi-dimensional tensor into one dimension.
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


torch.nn.Module.dump_patches = True
net = torch.load(r'./model/net_008.pth')
for i in range(20):
    # img = cv2.imread("D:/DeepLearning/digitalRecognition/digitalData/" + str(i) + ".bmp", 0)
    # shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # data = np.expand_dims(shrink, axis=2)
    # data = np.array(data, dtype=np.float32) / 255.0
    # data = np.expand_dims(data, axis=0)
    # # data = np.array(shrink, dtype=np.float32) / 255
    # data = data.reshape(1, 784)
    # negate784(data)
    image_path = r"D:/DeepLearning/digitalRecognition/digitalData/test/" + str(i) + ".bmp"
    img = cv2.imread(image_path, 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    data = np.reshape(shrink, 784)
    input_data = torch.empty(1, 784, dtype=torch.float)
    outputs = net(input_data)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)

    # print(str(i) + ' recognize result: ' + str(model.predict_classes(data, batch_size=1, verbose=0)[0]))
print(net)

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import cv2
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

torch.nn.Module.dump_patches = True
net = torch.load(r'./model/net_008.pth')
for i in range(20):
    numName = str(i)
    img = cv2.imread("D:/DeepLearning/digitalRecognition/digitalData/" + numName + ".bmp", 0)
    shrink = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    data = np.expand_dims(shrink, axis=2)
    data = np.array(data, dtype=np.float32) / 255.0
    data = np.expand_dims(data, axis=0)
    # data = np.array(shrink, dtype=np.float32) / 255
    data = data.reshape(1, 784)
    negate784(data)
    # print(str(i) + ' recognize result: ' + str(model.predict_classes(data, batch_size=1, verbose=0)[0]))
print(net)

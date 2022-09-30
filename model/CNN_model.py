# -*- coding: utf-8 -*-
# @Time     : 2022/9/23 16:27
# @Author   : Dyishere
# @File     : CNN_model.py
import sys
# from tkinter import S

sys.path.append("..")
from layer.conv2d import *
from layer.pool import *
from layer.linear import *
from layer.activation import *
import model
import time


class CNN_Net(model):
    def __init__(self, args):
        super().__init__()
        self.conv1 = Conv2d(1, 6, 5)
        self.conv2 = Conv2d(6, 16, 5)
        self.pool1 = MaxPool2d(2, 2)
        self.pool2 = MaxPool2d(2, 2)
        self.relu1 = Relu()
        self.relu2 = Relu()
        self.relu3 = Relu()
        self.linear1 = Linear(16 * 4 * 4, 120)
        self.linear2 = Linear(120, 10)
        self.lr = args.lr
        self.momentum = args.momentum

    def forward(self, x):
        # x.shape: batch_size, in_channels, in_height, in_weight
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        print("CNN x shape before:\n")
        print(x.shape)
        x = x.reshape(x.shape[0], -1)   # 转换为x.shape[0]行
        print("CNN x shape after:\n")
        print(x.shape)

        x = self.linear1.forward(x)
        x = self.relu3.forward(x)
        x = self.linear2.forward(x)
        t2 = time.time()

        return x

    def backpropagation(self, d_out):
        d_out = self.linear2.gradient(d_out)
        d_out = self.relu3.gradient(d_out)
        d_out = self.linear1.gradient(d_out)

        d_out = d_out.reshape((128, 16, 4, 4))

        d_out = self.pool2.gradient(d_out)
        d_out = self.relu2.gradient(d_out)
        d_out = self.conv2.gradient(d_out)

        d_out = self.pool1.gradient(d_out)
        d_out = self.relu1.gradient(d_out)
        d_out = self.conv1.gradient(d_out)

        print(self.conv1.w_grad[:2, :2, :3, :3])

        self.linear2.backward(self.lr, self.momentum)
        self.linear1.backward(self.lr, self.momentum)
        self.conv2.backward(self.lr, self.momentum)
        self.conv1.backward(self.lr, self.momentum)
        return

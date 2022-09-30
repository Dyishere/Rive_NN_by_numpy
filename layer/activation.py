# -*- coding: utf-8 -*-
# @Time     : 2022/9/22 16:21
# @Author   : Dyishere
# @File     : activation.py.py
from abc import ABC

import numpy as np


class softmax:
    def __init__(self, input_shape, axis):
        self.input_shape = input_shape
        self.axis = axis

    def forward(self, input_tensor):
        # input_tensor -= np.max(input_tensor, axis=self.axis)
        return np.exp(input_tensor) / np.sum(np.exp(input_tensor))

    def gradient(self, d_out):
        d_out = d_out.reshape(self.input_shape)
        return d_out


class Relu:
    def __init__(self):
        self.d_out = None
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, d_out):
        self.d_out = d_out
        self.d_out[self.x < 0] = 0
        return self.d_out


if __name__ == '__main__':
    # X = (np.arange(8) - 3).reshape(2, 4)
    # relu = Relu()
    # Y = relu.forward(X)
    # d_out = np.ones([2, 4])
    # g = relu.gradient(d_out)
    # print("X:", X)
    # print("Y:", Y)
    # print("g:", g)

    x = np.random.random((5, 256, 64, 64))
    y = softmax(x, axis=1)
    print(y.shape)

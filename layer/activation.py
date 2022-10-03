# -*- coding: utf-8 -*-
# @Time     : 2022/9/22 16:21
# @Author   : Dyishere
# @File     : activation.py.py
from abc import ABC

import numpy as np


class softmax:
    def __init__(self, input_shape, axis):
        self.out = None
        self.input_shape = input_shape
        self.axis = axis

    def forward(self, input_tensor):
        print(input_tensor.shape)
        if len(input_tensor.shape) == 4:
            max = np.zeros(input_tensor.shape)
            for i in range(max.shape[self.axis]):
                max[:, i, :, :] = np.max(input_tensor, axis=self.axis)
            input_tensor -= max
            return np.exp(input_tensor) / np.sum(np.exp(input_tensor))
        elif len(input_tensor.shape) == 3:
            max = np.zeros(input_tensor.shape)
            for i in range(max.shape[self.axis]):
                max[:, i, :] = np.max(input_tensor, axis=self.axis)
            input_tensor -= max
            self.out = np.exp(input_tensor) / np.sum(np.exp(input_tensor))
            return self.out
        else:
            print("Softmax 输入维度不支持！")
            return

    def gradient(self, d_out):
        dx = self.out * d_out
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out*sumdx
        # d_out = d_out.reshape(self.input_shape[:-1])
        # d_out = np.broadcast_to(d_out, shape=self.input_shape)
        return dx


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

    x = np.array(
        [0, 1, 2, 3, 4, 5]
    )
    y = softmax(x.shape, axis=0).forward(x)
    print(np.max(y))

# -*- coding: utf-8 -*-
# @Time     : 2022/9/23 14:47
# @Author   : Dyishere
# @File     : pool.py
import numpy as np


def zero_pad2d(X, pad):
    return np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # 环绕填充


class MaxPool2d:
    def __init__(self, K_size, stride=1, padding=0):
        self.out_W = None
        self.out_C = None
        self.out_H = None
        self.batch_size = None
        self.d_in = None
        self.K_size = K_size
        self.stride = stride
        self.pad = padding

    def forward(self, input_tensor):
        input_tensor = zero_pad2d(input_tensor, self.pad)
        batch_size, in_C, in_H, in_W, = input_tensor.shape  # batch_size, in_channels, in_height, in_weight

        self.d_in = np.zeros(input_tensor.shape)

        self.batch_size = batch_size
        self.out_H = int((in_H - self.K_size)/self.stride) + 1
        self.out_W = int((in_W - self.K_size)/self.stride) + 1
        self.out_C = in_C

        out_tensor = np.zeros((batch_size, self.out_C, self.out_H, self.out_W))

        for i in range(batch_size):
            one_input = input_tensor[i]
            for c in range(self.out_C):
                for h in range(self.out_H):
                    for w in range(self.out_W):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.K_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.K_size

                        input_slice = one_input[c, vert_start:vert_end, horiz_start:horiz_end]
                        out_tensor[i, c, h, w] = np.max(input_slice)
                        max_place = np.where(one_input[c, vert_start:vert_end, horiz_start:horiz_end] == out_tensor[i, c, h, w])
                        self.d_in[i, c, vert_start:vert_end, horiz_start:horiz_end][max_place] = 1

        return out_tensor

    def gradient(self, d_out):
        for i in range(self.batch_size):
            for c in range(self.out_C):
                for h in range(self.out_H):
                    for w in range(self.out_W):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.K_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.K_size
                        self.d_in[i, c, vert_start:vert_end, horiz_start:horiz_end] *= d_out[i, c, h, w]
        return self.d_in


if __name__ == '__main__':
    a = np.random.randint(1, 5, 64).reshape(2, 2, 4, 4)
    print(a)
    a = zero_pad2d(a, pad=1)
    print(a)
    # pool_max = MaxPool2d(2, 2)
    # out, _ = pool_max.forward(a)
    # print(out)
    # h = pool_max.gradient(np.ones(a.shape))
    # print(h)


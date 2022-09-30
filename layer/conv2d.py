# -*- coding: utf-8 -*-
# @Time     : 2022/9/22 16:31
# @Author   : Dyishere
# @File     : conv2d.py.py
import numpy as np


def zero_pad(X, pad):
    return np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # 环绕填充


def conv_calcu_step(input_slice, W, b):
    s = np.multiply(input_slice, W)
    out = np.sum(s)
    out += b
    return out


class Conv2d:
    def __init__(self, in_C, out_C, K_size, stride=1, padding=0):
        '''
        :param in_C: input Channel
        :param out_C: output Channel 相当于keras.layers.conv2d里的filter
        :param K_size:Kernel Size   卷积核 k*k
        :param stride:步长
        :param pad:padding
        '''
        self.out_W = None
        self.out_H = None
        self.batch_size = None
        self.x = None
        self.d_in = None

        a = np.sqrt(2) * np.sqrt(6 / (in_C + out_C)) / K_size  # a = √2 X √(6 / (in_C + out_C)) / K_size
        self.W = np.random.uniform(-a, a, size=(out_C, in_C, K_size, K_size))
        self.b = np.random.uniform(-a, a, size=(out_C, 1, 1, 1))
        self.stride = stride
        self.pad = padding

        # 核的梯度
        self.w_grad = np.zeros(self.W.shape)
        self.b_grad = np.zeros(self.b.shape)
        self.K_size = K_size
        self.out_C = out_C
        self.in_C = in_C

    def conv_calcu_step(input_slice, W, b):  # 单核运算
        s = input_slice * W
        out = np.sum(s)
        out += b
        return out

    def forward(self, x):
        self.x = x

        batch_size, in_C, in_H, in_W, = x.shape  # batch_size, in_channels, in_height, in_weight
        out_C, in_C, K_size, K_size = self.W.shape  # (in_C, K_size, K_size) 为一个卷积核

        self.batch_size = batch_size

        #  提前计算好长宽，避免后续循环中的越界判断，减少分支
        self.out_H = int((in_H + 2 * self.pad - K_size) / self.stride) + 1
        if in_H == in_W:
            self.out_W = self.out_H
        else:
            self.out_W = int((in_W + 2 * self.pad - K_size) / self.stride) + 1

        x = zero_pad(x, self.pad)
        out_tensor = np.zeros([batch_size, out_C, self.out_H, self.out_W])

        for i in range(batch_size):  # batch
            one_input = x[i]
            for c in range(out_C):  # Channel
                weights = self.W[c, :, :, :]
                biases = self.b[c, :, :, :]
                for h in range(self.out_H):  # Height
                    for w in range(self.out_W):  # Width
                        vert_start = h * self.stride  # 起始高
                        vert_end = vert_start + K_size  # 结束高
                        horiz_start = w * self.stride  # 起始宽
                        horiz_end = horiz_start + K_size  # 结束宽

                        input_slice = one_input[:, vert_start:vert_end, horiz_start:horiz_end]  # 全通道裁剪
                        out_tensor[i, c, h, w] = conv_calcu_step(input_slice, weights, biases)
        return out_tensor

    def gradient(self, d_out):
        x = zero_pad(self.x, self.pad)
        self.d_in = np.zeros(x.shape)
        for i in range(self.batch_size):
            one_input = x[i]
            for c in range(self.out_C):
                for h in range(self.out_H):
                    for w in range(self.out_W):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.K_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.K_size

                        input_slice = one_input[:, vert_start:vert_end, horiz_start:horiz_end]
                        self.w_grad[c] += input_slice * d_out[i, c, h, w]   # 求w的导数(链式求导)，y = w * x，则y' = x
                        self.b_grad[c] += d_out[i, c, h, w]     # dy/db = 1
                        self.d_in[i, :, vert_start:vert_end, horiz_start:horiz_end] += d_out[i, c, h, w] * self.W[c, :,
                                                                                                           :, :]

        if self.pad != 0:
            self.d_in = self.d_in[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return self.d_in

    def backward(self, alpha=0.001, momentum=0.9):
        self.W -= alpha * self.w_grad
        self.b -= alpha * self.b_grad
        # 零梯度
        self.w_grad *= momentum
        self.b_grad *= momentum
        return


if __name__ == '__main__':
    conv2d = Conv2d(in_C=3, out_C=64, K_size=3, padding=1)
    np.random.seed(11)
    X = np.random.randint(18, size=(5, 3, 16, 16))
    # w = np.random.uniform(-0.1, 0.1, (3, 2, 2, 2))
    # b = np.random.uniform(-0.1, 0.1, 3)
    # b = b.reshape(3, 1, 1, 1)
    # conv2d.W = w
    # conv2d.b = b

    print(conv2d.W)
    print(conv2d.b)

    Y = conv2d.forward(X)

    print(Y.shape)

    g = conv2d.gradient(np.ones(Y.shape))

    # print(conv2d.w_grad)
    # print(conv2d.b_grad)

    # print(g)

    conv2d.backward()

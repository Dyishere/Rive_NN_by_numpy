# -*- coding: utf-8 -*-
# @Time     : 2022/9/29 16:05
# @Author   : Dyishere
# @File     : batchNormalization.py
"""
参考博客：https://zhuanlan.zhihu.com/p/68685625
"""
import numpy as np


def broad_channels(matrix, axis, channels):
    matrix = np.broadcast_to(matrix, (channels,) + matrix.shape)
    temp_shape = np.hstack(
        (np.arange(1, axis + 1), (0,))
    )
    temp_shape = np.hstack(
        (temp_shape, np.arange(axis + 1, len(matrix.shape)))
    )
    matrix = matrix.transpose(temp_shape)
    return matrix


class BatchNorm:
    def __init__(self, in_dim, affine=True, eps=1e-8, momentum=0.1):
        self.std = None
        self.V = None
        self.u = None
        self.gamma_grad = 0.
        self.beta_grad = 0.
        self.x = None
        self.eps = eps
        self.gamma = np.random.uniform(0.9, 1.1, in_dim)
        self.beta = np.random.uniform(-0.1, 0.1, in_dim)
        self.gamma_s = None
        self.y = None
        self.momentum = momentum

        self.affine = affine

    def forward(self, input_tensor):
        self.x = input_tensor
        self.u = input_tensor.mean(axis=0)   # 均值
        self.V = input_tensor.var(axis=0)    # 方差

        # 标准差std = √(V + ε), ε是个极小值（如1e-8）防止方差为0
        self.std = np.sqrt(self.V + self.eps)

        # 更新可学习参数γ
        self.gamma_s = self.gamma / self.std
        self.y = (self.x - self.u) / self.std
        return self.gamma * self.y + self.beta

    def gradient(self, d_out):
        if not self.affine:                  # 如果是对输入层做归一化，就不做向上传播梯度
            return
        batch_size = d_out.shape[0]
        gamma = self.gamma
        self.beta_grad += np.sum(d_out, axis=0)
        self.gamma_grad += np.sum(d_out * (self.x - self.u) * self.std, axis=0)

        d_out = (1/batch_size) * gamma * self.std * (
            batch_size * d_out
            - self.beta_grad
            - self.y * self.gamma_grad
        )
        return d_out

    def backward(self, alpha=0.001, momentum=0.9):
        if not self.affine:                  # 如果是对输入层做归一化，就不做向上传播梯度
            return
        if self.momentum is not None:
            momentum = self.momentum
        self.gamma -= self.gamma_grad * alpha
        self.beta -= self.beta_grad * alpha

        self.gamma_grad *= momentum
        self.beta_grad *= momentum
        return


class BatchNorm2d:
    def __init__(self, in_dim, affine=True, eps=1e-8, momentum=0.1, axis=1):
        self.beta_b = None
        self.gamma_b = None
        self.std = None
        self.V = None
        self.u = None
        self.gamma_grad = 0.
        self.beta_grad = 0.
        self.x = None
        self.eps = eps
        self.gamma = np.random.uniform(0.9, 1.1, np.hstack((in_dim[:axis], in_dim[axis+1:])))
        self.beta = np.random.uniform(-0.1, 0.1, np.hstack((in_dim[:axis], in_dim[axis+1:])))
        self.gamma_s = None
        self.y = None
        self.momentum = momentum

        self.affine = affine
        self.axis = axis
        self.channels = in_dim[axis]

    def forward(self, input_tensor):
        self.x = input_tensor
        self.u = input_tensor.mean(axis=self.axis)   # 均值
        self.V = input_tensor.var(axis=self.axis)    # 方差

        # 标准差std = √(V + ε), ε是个极小值（如1e-8）防止方差为0
        self.std = np.sqrt(self.V + self.eps)

        # 更新可学习参数γ
        self.gamma_s = self.gamma / self.std

        # 增广u、V、std
        self.u = broad_channels(self.u, self.axis, self.channels)
        # self.V = broad_channels(self.V, self.axis, self.channels)
        self.std = broad_channels(self.std, self.axis, self.channels)
        self.gamma_b = broad_channels(self.gamma, self.axis, self.channels)
        self.beta_b = broad_channels(self.beta, self.axis, self.channels)

        self.y = (self.x - self.u) / self.std
        return self.gamma_b * self.y + self.beta_b

    def gradient(self, d_out):
        if not self.affine:                  # 如果是对输入层做归一化，就不做向上传播梯度
            return
        batch_size = d_out.shape[0]
        gamma = self.gamma
        self.beta_grad += np.sum(d_out, axis=self.axis)
        self.gamma_grad += np.sum(d_out * (self.x - self.u) * self.std, axis=self.axis)

        gamma_grad_b = broad_channels(self.gamma_grad, self.axis, self.channels)
        beta_grad_b = broad_channels(self.beta_grad, self.axis, self.channels)

        d_out = (1/batch_size) * self.gamma_b * self.std * (
            batch_size * d_out
            - beta_grad_b
            - self.y * gamma_grad_b
        )
        return d_out

    def backward(self, alpha=0.001, momentum=0.9):
        if not self.affine:                  # 如果是对输入层做归一化，就不做向上传播梯度
            return
        if self.momentum is not None:
            momentum = self.momentum
        self.gamma -= self.gamma_grad * alpha
        self.beta -= self.beta_grad * alpha

        self.gamma_grad *= momentum
        self.beta_grad *= momentum
        return


if __name__ == '__main__':
    x = np.random.random((5, 64, 16, 16))
    bn = BatchNorm2d(in_dim=x.shape, axis=1)
    y = bn.forward(x)
    # print(y.shape)
    # print(bn.gradient(x))
    bn.backward()
    # x = np.random.random((2, 4, 4))
    # y = np.broadcast_to(x, (3,)+x.shape)
    # print(y.shape)
    # y = y.transpose((1, 0, 2, 3))
    # if np.equal(y[:, 0, :, :].all(), y[:, 1, :, :].all()):
    #     print("Yes!")
    # print()
    # print(y[:, 0])
    # print(y[:, 1])
    # print(y.shape)

    # x = (0, 1, 2, 3, 4, 5)
    # axis = 2
    # y = np.hstack((x[axis], x[0:axis]))
    # z = np.hstack((y, x[axis+1:]))
    # print(z)

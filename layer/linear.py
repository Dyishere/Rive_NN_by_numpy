# -*- coding: utf-8 -*-
# @Time     : 2022/9/23 16:35
# @Author   : Dyishere
# @File     : linear.py
import numpy as np


class Linear:
    def __init__(self, in_dim, out_dim):
        self.x = None
        self.in_dim = in_dim
        self.out_dim = out_dim

        a = np.sqrt(2) * np.sqrt(6/(in_dim + out_dim))

        self.W = np.random.uniform(-a, a, (in_dim, out_dim))
        self.b = np.random.uniform(-a, a, out_dim)

        self.w_grad = np.zeros(self.W.shape)
        self.b_grad = np.zeros(self.b.shape)

    def forward(self, input_tensor):
        self.x = input_tensor
        output = np.dot(input_tensor, self.W)+self.b
        return output

    def gradient(self, d_out):
        batch_size = d_out.shape[0]
        for i in range(batch_size):
            col_x = self.x[i][:, np.newaxis]
            d_out_i = d_out[i][:, np.newaxis].T
            self.w_grad += np.dot(col_x, d_out_i)
            self.b_grad += d_out_i.reshape(self.b.shape)

        d_in = np.dot(d_out, self.W.T)
        d_in = np.reshape(d_in, self.x.shape)

        return d_in

    def backward(self, alpha=0.001, momentum=0.9):
        self.W -= alpha * self.w_grad
        self.b -= alpha * self.b_grad

        self.w_grad *= momentum
        self.b_grad *= momentum
        return


if __name__ == '__main__':
    img = np.array([[1, 2, 3], [8, 7, 6]])
    fc = Linear(3, 5)
    fc.W = np.array([[-0.0942, 0.2644, -0.5415],
                     [0.5192, -0.3060, 0.1294],
                     [0.2440, 0.3442, 0.5114],
                     [-0.4632, 0.3757, -0.2355],
                     [0.2432, -0.5362, 0.1867]]).T
    fc.b = np.array([0.0494, 0.3470, -0.4474, -0.3366, 0.3447])
    out = fc.forward(img) * 0.1
    print(out)
    d_in = fc.gradient(np.ones(out.shape))
    print(d_in)

    print(fc.w_grad)
    print(fc.b_grad)

    fc.backward()
    print(fc.W)


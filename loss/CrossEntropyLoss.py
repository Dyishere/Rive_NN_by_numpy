# -*- coding: utf-8 -*-
# @Time     : 2022/9/23 15:33
# @Author   : Dyishere
# @File     : CrossEntropyLoss.py
"""
功能描述：
交叉熵
"""
import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.d_out = None
        self.batch_size = None
        self.soft_X = None
        self.X = None
        self.loss = None
        self.labels = None
    
    def cal_loss(self, X, y):
        self.labels = y
        self.softmax(X)
        self.loss = 0
        for i in range(self.batch_size):
            self.loss += -np.log(self.soft_X[i, int(y[i])]) / self.batch_size
        return self.loss

    def softmax(self, X):
        self.X = X
        self.soft_X = np.zeros(X.shape)
        self.batch_size = X.shape[0]
        for i in range(self.batch_size):
            self.X[i] -= np.max(X[i])
            self.soft_X[i] = np.exp(X[i]) / np.sum(np.exp(X[i]))    # 归一化
        return self.soft_X

    def gradient(self):
        self.d_out = self.soft_X.copy()
        for i in range(self.batch_size):
            self.d_out[i, int(self.labels[i])] -= 1
        return self.d_out / self.batch_size

        
if __name__ == '__main__':
    criterion = CrossEntropyLoss()
    X = np.array([[0, 1, 2], [3, 4, 5]])
    print(X)
    y = np.array([[1], [2]])
    print(y)
    loss = criterion.cal_loss(X, y)
    print(loss)
    print(criterion.soft_X)
    print(criterion.gradient())

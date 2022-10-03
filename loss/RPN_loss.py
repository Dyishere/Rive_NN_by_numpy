# -*- coding: utf-8 -*-
# @Time     : 2022/10/1 15:57
# @Author   : Dyishere
# @File     : RPN_loss.py
"""
功能描述：

"""
import numpy as np
from loss.CrossEntropyLoss import *


def batch_pack(x, batch_counts):
    output = []
    for i in range(len(batch_counts)):
        for j in range(batch_counts[i]):
            output.append(x[i, j, :])
    return output


class rpn_reg_loss:
    def __init__(self):
        self.d_out = None
        self.L1_loss = None
        self.target_bbox = None
        self.batch_size = None
        self.rpn_bbox = None
        self.indices = None
        self.rpn_match = None

    def forward(self, rpn_match, input_rpn_bbox, rpn_bbox):
        """

        :param rpn_match:(batch_size, 576, 1)
        :param input_rpn_bbox:bbox偏移值真实值 (batch_size, 576, 4)
        :param rpn_bbox:bbox偏移值预测值 (batch_size, 576, 4)
        :return:
        """
        self.d_out = np.zeros(shape=rpn_bbox.shape)
        self.batch_size = rpn_match.shape[0]
        # _____________________________________
        # 计算回归损失
        self.rpn_match = np.squeeze(rpn_match)
        # rpn_match.shape = (None, 576)

        # 取出label等于1的坐标,抛弃label等于0或-1的无效元素，因为我们只需要学习正确的bbox的修正值
        self.indices = np.argwhere(self.rpn_match == 1)
        # indices.shape=(sum(label=1), 2)

        # 提取rpn_bbox中索引为[indices]的向量
        self.rpn_bbox = rpn_bbox[self.indices[:, 0], self.indices[:, 1], :]  # prediction
        # print(self.rpn_bbox.shape)
        # prediction shape = (sum(label=1), 4)

        # 计算出每个batch有几个有效anchor
        batch_counts = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            batch_counts[i] = len(np.argwhere(self.indices[:, 0] == i))
        batch_counts = batch_counts.astype(int)

        # 按batch_counts取出target_bbox
        self.target_bbox = batch_pack(input_rpn_bbox, batch_counts)
        self.target_bbox = np.array(self.target_bbox)
        # self.target_bbox shape (batch_size, batch_counts[batch_size], 4)

        # 求出L1 loss
        assert self.target_bbox.shape == self.rpn_bbox.shape
        self.L1_loss = np.abs(self.target_bbox - self.rpn_bbox)

        for i in range(self.L1_loss.shape[0]):
            for j in range(self.L1_loss.shape[1]):
                self.L1_loss[i, j] = 0.5 * (self.L1_loss[i, j] - int(self.L1_loss[i, j]))**2 + \
                                     float(int(self.L1_loss[i, j])) - 0.5

        return self.L1_loss

    def backpropagation(self, d_out):
        for i in range(self.indices.shape[0]):
            self.d_out[self.indices[i, 0], self.indices[i, 1], :] = self.L1_loss[i, :]

        return self.d_out


class rpn_cls_loss:
    def __init__(self):
        self.d_out = None
        self.cls_loss = None
        self.criterion = None
        self.y_true = None
        self.anchor_class = None
        self.y_pred = None
        self.indices = None
        self.rpn_match = None

    def forward(self, rpn_match, rpn_class_logits):
        # 反向传播预留参数
        self.d_out = np.zeros(shape=rpn_class_logits.shape)

        # _____________________________________
        # 计算分类损失
        # rpn_match (batch_size, 576, 1)            每个锚框对应的*真实*label
        # rpn_class_logits (batch_size, 576, 2)     RPN网络实际计算结果
        self.rpn_match = np.squeeze(rpn_match)
        # print("rpn_match.shape = {}".format(rpn_match.shape))
        # rpn_match.shape = (None, 576)

        # 取出label等于-1和1的坐标,抛弃label等于0的无效元素
        self.indices = np.argwhere(self.rpn_match != 0)
        # print("indices.shape = {}".format(self.indices.shape))
        # indices.shape=(sum(label=1|label=-1), 2)

        # 提取rpn_class_logits中索引为[indices]的向量
        self.y_pred = rpn_class_logits[self.indices[:, 0], self.indices[:, 1], :]  # prediction
        # prediction shape = (sum(label=1|label=-1), 2)

        self.anchor_class = self.rpn_match[self.indices[:, 0], self.indices[:, 1]]  # target 对应
        self.y_true = (self.anchor_class + 1) / 2

        # 引入交叉熵损失层
        self.criterion = CrossEntropyLoss()
        self.cls_loss = self.criterion.cal_loss(X=self.y_pred, y=self.y_true)
        return self.cls_loss

    def backpropagation(self):
        y = self.criterion.gradient()
        for i in range(self.indices.shape[0]):
            self.d_out[self.indices[:, 0], self.indices[:, 1], :] = y[i, :]
        return self.d_out


if __name__ == '__main__':
    match = np.random.randint(-1, 2, (3, 576, 1))
    cls = np.random.random((3, 576, 2))
    input_rpn_bbox = np.random.random((3, 576, 4))
    rpn_bbox = np.random.random((3, 576, 4))
    rpn_reg = rpn_reg_loss()
    d_out1 = rpn_reg.forward(match, input_rpn_bbox, rpn_bbox)

    rpn_cls = rpn_cls_loss()
    d_out2 = rpn_cls.forward(rpn_match=match, rpn_class_logits=cls)
    print(d_out1.shape)
    print(d_out2)
    print(rpn_reg.backpropagation(d_out1).shape)
    print(rpn_cls.backpropagation()[0, 0, :])

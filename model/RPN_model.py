# -*- coding: utf-8 -*-
# @Time     : 2022/10/1 1:14
# @Author   : Dyishere
# @File     : RPN_model.py
"""
功能描述：

"""
import sys

import numpy as np

sys.path.append("..")
import model
from layer.conv2d import *
from layer.pool import *
from layer.linear import *
from layer.activation import *
from layer.batchNormalization import *
from layer.conv2d import Conv2d as conv2d
import time


class RPN:
    def __init__(self, input_shape, anchor_num):
        """

        :param input_shape:Feature Map, shape = (batch_size, 1024, 8, 8)
        :param anchor_num:
        """
        self.bbox_shape = None
        self.cls_shape = None
        self.shared_map = conv2d(in_C=input_shape[1], out_C=256, K_size=3, padding=1)
        # shared_map shape = (batch_size, 256, 8, 8)
        self.conv_cls = conv2d(in_C=256, out_C=2*anchor_num, K_size=1)
        # conv_cls shape = (batch_size, 2*anchor_num, 8, 8)
        self.input_shape = list(input_shape)
        self.input_shape[1] = 2*anchor_num
        self.input_shape = tuple(self.input_shape)
        self.softmax = softmax(input_shape=self.input_shape, axis=1)
        # reshape(batch_size, anchor_num, 2)
        # softmax()
        self.conv_rec = conv2d(in_C=256, out_C=4*anchor_num, K_size=1)
        # reshape(batch_size, anchor_num, 4)

    def forward(self, input_tensor):
        y = self.shared_map.forward(input_tensor)
        cls = self.conv_cls.forward(y)
        cls = cls.transpose((0, 2, 3, 1))
        self.cls_shape = cls.shape
        cls = cls.reshape((cls.shape[0], -1, 2))
        prob = self.softmax.forward(cls)

        bbox = self.conv_rec.forward(y)
        self.bbox_shape = bbox.shape
        bbox = bbox.reshape((bbox.shape[0], -1, 4))

        return cls, prob, bbox

    def backpropagation(self, d_out):
        cls_loss, prob_loss, bbox_loss = d_out[0], d_out[1], d_out[2]
        bbox_loss = bbox_loss.reshape(self.bbox_shape)
        bbox_loss = self.conv_rec.gradient(bbox_loss)
        self.conv_rec.backward()

        cls_loss = self.softmax.gradient(prob_loss)
        cls_loss = cls_loss.reshape(self.cls_shape).transpose((0, 3, 1, 2))
        cls_loss = self.conv_cls.gradient(cls_loss)
        self.conv_cls.backward()

        d_out = self.shared_map.gradient(cls_loss+bbox_loss)
        self.shared_map.backward()
        return d_out


if __name__ == '__main__':
    rpn = RPN(input_shape=(5, 1024, 8, 8), anchor_num=9)
    x = np.random.random((5, 1024, 8, 8))
    cls, prob, bbox = rpn.forward(input_tensor=x)
    print(cls.shape)
    print(prob.shape)
    print(bbox.shape)
    print(rpn.backpropagation((cls, prob, bbox)))


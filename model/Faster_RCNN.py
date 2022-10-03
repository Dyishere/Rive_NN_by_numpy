# -*- coding: utf-8 -*-
# @Time     : 2022/9/29 20:36
# @Author   : Dyishere
# @File     : Faster_RCNN.py
"""
功能描述：

"""

import sys
sys.path.append("..")
from layer.conv2d import *
from layer.pool import *
from layer.linear import *
from layer.activation import *
from layer.batchNormalization import *
import time
import numpy as np

from ResNet_model import *
from RPN_model import *
from loss.RPN_loss import *


class Faster_RCNN:
    def __init__(self, batch_size, image_size, anchor_num):
        self.reg_loss = None
        self.cls_loss = None
        self.rpn_bbox = None
        self.rpn_prob = None
        self.rpn_cls = None
        self.batch_size = batch_size
        self.input_image_shape = (self.batch_size, 3, image_size, image_size)
        self.input_bboxes = (anchor_num, 4)
        self.input_cls_label = ()

        # 模型初始化
        self.resNet = resNet_featureExtractor(self.input_image_shape)
        self.rpnNet = RPN(input_shape=(self.batch_size, 1024, image_size, image_size), anchor_num=anchor_num)

        # loss层初始化
        self.rpn_cls_loss = rpn_cls_loss()
        self.rpn_reg_loss = rpn_reg_loss()

    def forward(self, input_tensor, input_bboxes, input_cls_label, input_rpn_match, input_rpn_bbox):
        # x.shape: batch_size, in_channels, in_height, in_weight
        featureMap = self.resNet.forward(input_tensor)
        self.rpn_cls, self.rpn_prob, self.rpn_bbox = self.rpnNet.forward(featureMap)
        self.cls_loss = self.rpn_cls_loss.forward(input_cls_label, self.rpn_prob)
        self.reg_loss = self.rpn_reg_loss.forward(input_rpn_match, input_rpn_bbox, self.rpn_bbox)
        return self.cls_loss, self.reg_loss

    def backpropagation(self, d_out):
        cls_loss = self.rpn_cls_loss.backpropagation()
        reg_loss = self.rpn_reg_loss.backpropagation(d_out)

        d_out = self.rpnNet.backpropagation([cls_loss, reg_loss])
        d_out = self.resNet.backpropagation(d_out)
        return d_out


if __name__ == '__main__':
    # 一个batch
    # input_image (3, 256, 256)
    # input_bboxes (bbox个数, 4)
    # input_cls_label (bbox个数, 1), dtype=int32                 真实值
    # input_rpn_match (bbox个数, 1), dtype=int32 {-1, 0, 1}   预测值
    # input_rpn_bbox (bbox个数, 4)
    epoch_num = 5
    batch_size = 3
    input_image_size = 64
    FeatureMap_image_size = int(input_image_size / 8)

    # Anchor Scale
    # Scales = [2, 4, 8]
    Scales = [1, 2, 4]
    # Anchor Shape
    Ratios = [0.5, 1, 2]
    anchor_num = int(FeatureMap_image_size*FeatureMap_image_size*len(Scales) * len(Ratios))
    batch_input_image = np.random.random((batch_size, 3, input_image_size, input_image_size))
    batch_input_bboxes = np.random.random((batch_size, len(Scales) * len(Ratios), 4))
    input_cls_label = np.random.randint(-1, 2, (batch_size, anchor_num, 1))
    input_rpn_match = np.random.randint(-1, 2, (batch_size, anchor_num, 1))
    input_rpn_bbox = np.random.random((batch_size, anchor_num, 4))

    model = Faster_RCNN(batch_size=batch_size, image_size=input_image_size, anchor_num=len(Scales) * len(Ratios))

    y = model.forward(batch_input_image, batch_input_bboxes, input_cls_label, input_rpn_match, input_rpn_bbox)
    print("y:{}".format(y))

    print(model.backpropagation(y).shape)


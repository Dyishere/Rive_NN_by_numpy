# -*- coding: utf-8 -*-
# @Time     : 2022/9/30 15:16
# @Author   : Dyishere
# @File     : ResNet_model.py
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


class building_block:
    def __init__(self, input_shape, filters, block, model_name='Building_block'):
        """

        :param input_shape: (batch_size, Channels, Height, Width)
        :param filters: output Channels = 4*filters
        :param block:
        """
        self.name = model_name
        self.block = block
        self.stride = 1
        if block != 0:
            self.stride = 1     # 保持Channels=256
        else:
            self.stride = 2     # 两个维度均下采样两倍 64=256/4，to 64
        # input X
        self.input_shape = input_shape
        self.input_shape_forward = input_shape

        self.conv1 = conv2d(in_C=self.input_shape[1], out_C=filters, K_size=1, stride=self.stride)
        self.input_shape = list(self.input_shape)
        self.input_shape[1] = filters
        self.input_shape[2] = int(self.input_shape[2]/self.stride)
        self.input_shape[3] = int(self.input_shape[3]/self.stride)
        self.input_shape = tuple(self.input_shape)
        self.bn1 = BatchNorm2d(in_dim=self.input_shape, eps=1e-05, momentum=0.1, affine=True, axis=1)
        self.Activation1 = Relu()

        # padding='same', out_size = [in_size/stride]向上取整
        self.conv2 = conv2d(in_C=self.input_shape[1], out_C=filters, K_size=3, padding=1)
        self.input_shape = list(self.input_shape)
        self.input_shape[1] = filters
        self.input_shape = tuple(self.input_shape)
        self.bn2 = BatchNorm2d(in_dim=self.input_shape, eps=1e-05, momentum=0.1, affine=True)
        self.Activation2 = Relu()

        self.conv3 = conv2d(in_C=self.input_shape[1], out_C=4*filters, K_size=1)
        self.input_shape = list(self.input_shape)
        self.input_shape[1] = 4*filters
        self.input_shape = tuple(self.input_shape)
        self.bn3 = BatchNorm2d(in_dim=self.input_shape, eps=1e-05, momentum=0.1, affine=True)
        # output Y
        if block == 0:
            # output shortcut = shortcut(input X)
            self.shortcut_conv1 = conv2d(in_C=input_shape[1], out_C=4*filters, K_size=1, stride=self.stride)
            self.input_shape = list(input_shape)
            self.input_shape[1] = 4*filters
            self.input_shape[2] = int(self.input_shape[2] / self.stride)
            self.input_shape[3] = int(self.input_shape[3] / self.stride)
            self.input_shape = tuple(self.input_shape)
            self.shortcut_norm1 = BatchNorm2d(in_dim=self.input_shape, eps=1e-05, momentum=0.1, affine=True)
        else:
            # output shortcut = input X
            pass

        # output Y = input Y + input X
        self.Activation3 = Relu()
        # return Y

    def forward(self, x):
        assert x.shape == self.input_shape_forward, '[{0}]:输入尺度与模型构造不符 输入：{1} 预测：{2}'.format(
            self.name, x.shape, self.input_shape_forward
        )
        # print("x:{}".format(x.shape))
        y = self.conv1.forward(x)
        # print("y1:{}".format(y.shape))
        y = self.bn1.forward(y)
        # print("y2:{}".format(y.shape))
        y = self.Activation1.forward(y)
        # print("y3:{}".format(y.shape))

        y = self.conv2.forward(y)
        # print("y4:{}".format(y.shape))
        y = self.bn2.forward(y)
        # print("y5:{}".format(y.shape))
        y = self.Activation2.forward(y)
        # print("y6:{}".format(y.shape))

        y = self.conv3.forward(y)
        y = self.bn3.forward(y)
        # print("y1:{}".format(y.shape))

        if self.block == 0:
            # print("shortcut_input:{}".format(x.shape))
            shortcut = self.shortcut_conv1.forward(x)
            # print("shortcut_conv2d:{}".format(shortcut.shape))
            shortcut = self.shortcut_norm1.forward(shortcut)
            # print("shortcut_conv2d_outpust:{}".format(shortcut.shape))
        else:
            shortcut = x
        y = np.add(y, shortcut)
        y = self.Activation3.forward(y)
        return y

    def backpropagation(self, d_out):
        d_out = self.Activation3.gradient(d_out)
        if self.block == 0:
            shortcut = self.shortcut_norm1.gradient(d_out)
            self.shortcut_norm1.backward()
            # print("d_out1:{}".format(d_out.shape))
            shortcut = self.shortcut_conv1.gradient(shortcut)
            self.shortcut_conv1.backward()
            # print("shortcut:{}".format(d_out.shape))
        else:
            pass
        d_out = self.bn3.gradient(d_out)
        self.bn3.backward()
        d_out = self.conv3.gradient(d_out)
        self.conv3.backward()

        d_out = self.Activation2.gradient(d_out)
        d_out = self.bn2.gradient(d_out)
        self.bn2.backward()
        d_out = self.conv2.gradient(d_out)
        self.conv2.backward()

        d_out = self.Activation1.gradient(d_out)
        d_out = self.bn1.gradient(d_out)
        self.bn1.backward()
        d_out = self.conv1.gradient(d_out)
        self.conv1.backward()
        return d_out


class resNet_featureExtractor:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self.conv1 = conv2d(in_C=3, out_C=64, K_size=3, padding=1)
        self.input_shape = list(self.input_shape)
        self.input_shape[1] = 64
        self.input_shape = tuple(self.input_shape)
        self.bn1 = BatchNorm2d(in_dim=self.input_shape, eps=1e-05, momentum=0.1, affine=True, axis=1)
        self.Activation1 = Relu()

        filters = 64
        blocks = [3, 6, 4]

        # print("预期输入：{}".format(self.input_shape))
        self.building_block_00 = building_block(input_shape=self.input_shape, filters=filters, block=0)
        self.input_shape = list(self.input_shape)
        self.input_shape[1] = 4 * filters
        self.input_shape[2] = int(self.input_shape[2] / 2)
        self.input_shape[3] = int(self.input_shape[3] / 2)
        self.input_shape = tuple(self.input_shape)
        self.building_block_01 = building_block(input_shape=self.input_shape, filters=filters, block=1)
        self.building_block_02 = building_block(input_shape=self.input_shape, filters=filters, block=2)

        filters *= 2
        self.building_block_10 = building_block(input_shape=self.input_shape, filters=filters, block=0)
        self.input_shape = list(self.input_shape)
        self.input_shape[1] = 4 * filters
        self.input_shape[2] = int(self.input_shape[2] / 2)
        self.input_shape[3] = int(self.input_shape[3] / 2)
        self.input_shape = tuple(self.input_shape)
        self.building_block_11 = building_block(input_shape=self.input_shape, filters=filters, block=1)
        self.building_block_12 = building_block(input_shape=self.input_shape, filters=filters, block=2)
        self.building_block_13 = building_block(input_shape=self.input_shape, filters=filters, block=3)
        self.building_block_14 = building_block(input_shape=self.input_shape, filters=filters, block=4)
        self.building_block_15 = building_block(input_shape=self.input_shape, filters=filters, block=5)

        filters *= 2
        self.building_block_20 = building_block(input_shape=self.input_shape, filters=filters, block=0)
        self.input_shape = list(self.input_shape)
        self.input_shape[1] = 4 * filters
        self.input_shape[2] = int(self.input_shape[2] / 2)
        self.input_shape[3] = int(self.input_shape[3] / 2)
        self.input_shape = tuple(self.input_shape)
        self.building_block_21 = building_block(input_shape=self.input_shape, filters=filters, block=1)
        self.building_block_22 = building_block(input_shape=self.input_shape, filters=filters, block=2)
        self.building_block_23 = building_block(input_shape=self.input_shape, filters=filters, block=3)

    def forward(self, input_tensor):
        x = self.conv1.forward(input_tensor)
        x = self.bn1.forward(x)
        x = self.Activation1.forward(x)
        # print("实际输入：{}".format(x.shape))
        x = self.building_block_00.forward(x)
        x = self.building_block_01.forward(x)
        x = self.building_block_02.forward(x)

        x = self.building_block_10.forward(x)
        x = self.building_block_11.forward(x)
        x = self.building_block_12.forward(x)
        x = self.building_block_13.forward(x)
        x = self.building_block_14.forward(x)
        x = self.building_block_15.forward(x)

        x = self.building_block_20.forward(x)
        x = self.building_block_21.forward(x)
        x = self.building_block_22.forward(x)
        x = self.building_block_23.forward(x)

        return x

    def backpropagation(self, d_out):
        # print("d_out:{}".format(d_out.shape))
        y = self.building_block_23.backpropagation(d_out)
        # print("y1:{}".format(y.shape))
        y = self.building_block_22.backpropagation(y)
        # print("y2:{}".format(y.shape))
        y = self.building_block_21.backpropagation(y)
        # print("y3:{}".format(y.shape))
        y = self.building_block_20.backpropagation(y)
        # print("y4:{}".format(y.shape))

        y = self.building_block_15.backpropagation(y)
        y = self.building_block_14.backpropagation(y)
        y = self.building_block_13.backpropagation(y)
        y = self.building_block_12.backpropagation(y)
        y = self.building_block_11.backpropagation(y)
        y = self.building_block_10.backpropagation(y)

        y = self.building_block_02.backpropagation(y)
        y = self.building_block_01.backpropagation(y)
        y = self.building_block_00.backpropagation(y)

        y = self.Activation1.gradient(y)
        y = self.bn1.gradient(y)
        self.bn1.backward()
        y = self.conv1.gradient(y)
        self.conv1.backward()

        return y


if __name__ == '__main__':
    x = np.random.random((5, 3, 64, 64))
    bb = resNet_featureExtractor(input_shape=x.shape)
    y = bb.forward(x)
    print("out Shape:{}".format(y.shape))
    print("reverse input Shape:{}".format(bb.backpropagation(y).shape))





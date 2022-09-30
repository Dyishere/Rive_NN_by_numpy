# -*- coding: utf-8 -*-
# @Time     : 2022/9/22 16:27
# @Author   : Dyishere
# @File     : main.py.py
"""
功能描述：

"""
import numpy as np
from model.ResNet_model import *
from model.RPN_model import *

if __name__ == '__main__':
    x = np.random.random((5, 3, 64, 64))
    resnet = resNet_featureExtractor(x.shape)
    rpn = RPN(input_shape=(5, 1024, 8, 8), anchor_num=9)
    y = resnet.forward(x)
    y = rpn.forward(y)
    print(y)
    d = rpn.backpropagation(y)
    d = resnet.backpropagation(d)
    print(d)


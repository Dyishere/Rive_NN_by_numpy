# -*- coding: utf-8 -*-
# @Time     : 2022/9/29 16:49
# @Author   : Dyishere
# @File     : layer.py
"""
功能描述：
    所有层的基类
"""
from abc import ABCMeta, abstractmethod


class layer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass


# -*- coding: utf-8 -*-
# @Time     : 2022/9/29 16:49
# @Author   : Dyishere
# @File     : layer.py
"""
功能描述：
    所有模型的基类
"""
from abc import ABCMeta, abstractmethod


class model(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backpropagation(self, *args):
        pass


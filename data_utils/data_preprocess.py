# -*- coding: utf-8 -*-
# @Time     : 2022/9/24 14:31
# @Author   : Dyishere
# @File     : data_preprocess.py
from random import random
from mlxtend.data import loadlocal_mnist
import numpy as np
import gzip
import matplotlib.pyplot as plt
import random
import argparse


def extract_data(image_path, label_path):
    X, y = loadlocal_mnist(
        images_path=image_path,
        labels_path=label_path
    )
    return X, y

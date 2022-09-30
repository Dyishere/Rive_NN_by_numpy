# -*- coding: utf-8 -*-
# @Time     : 2022/9/29 9:21
# @Author   : Dyishere
# @File     : anchor_generator.py
"""
全称：Region Proposal Network

功能描述：
    用深度学习的办法从图像中提取候选框

1.从属 Faster-RCNN 网络结构：
    [Image] (None, N, N, 3)
    -to
    <CNN>
    抽取 Feature Map
    -to
        [Feature Map] (None, n, n, d)
    -to                         -to
    <Slide> & <FC>              --to        # Sliding: 3x3 卷积核滑动，相当于卷积
    卷积核：3x3x256，256个3x3的卷积核
    -to                         ---to
    抽取[Feature Map]上每一个点的特征信息        # Feature Map上的一个点大多对应Image上的一个区域
    -to
    <*RPN>                      ----to
    -to                         -----to
    [proposals]                 ------to
    映射到Image上，确定锚框
    -to                         -------to
        <RoI Pooling>
    -to
        <classifier>

2.RPN网络
    <Slide> & <FC>
    -to
    [Shared Map] (None, n, n, 256)                  # nxn个特征向量，每个特征向量有256个维度
    -to                         -to
    1x1 Conv(1*1*2k)            1x1 Conv(1*1*4k)    # 相当于全连接
    k:Feature Map上点与锚框的个数
    2:(前景概率,背景概率)          4:(x,y,w,h)左上角坐标
    -to                         -to
    [cls]                       [reg]

    [PS]RPN网络的输出信息
    -特征信息
        -分类信息
            --前景概率（提取前景，确定锚框）
            --背景概率
        -回归信息
            --修正（逼近bbox）


3.对比：Fast-RCNN 网络结构：
    <Image algorithm: 如 Selective Search>提取候选框
    -to
    [Image] & [ROI]
    -to             -to
    <Deep ConvNet>  <RoI Projection>
    提取特征          映射RoI
    -to             -to
        <Conv Feature Map> with [RoI]
    -to
    <RoI Pooling Layer>
    尺寸标准化
    -to
    <FCs>
    -to
    [RoI Feature Vector]
    -to             -to
    <FC>            <FC>
    -to             -to
    <Softmax>       <bbox Regressor>
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Feature Map Size 16x16
Size_X = 16
Size_Y = 16

# Image to Feature Map Scale
rpn_Stride = 8
# Image Size
image_X = Size_X * rpn_Stride
image_Y = Size_Y * rpn_Stride

# Anchor Scale
# Scales = [2, 4, 8]
Scales = [1, 2, 4]
# Anchor Shape
Ratios = [0.5, 1, 2]


def anchor_generator(size_X, size_Y, rpn_stride, scales, ratios):
    """

    :param size_X: int
    :param size_Y: int
    :param rpn_stride: int
    :param scales: list
    :param ratios: list
    :return: (Scales.shape * Ratios.shape * Size_X * Size_Y, 2)
    """
    # 生成锚框参数矩阵
    scales, ratios = np.meshgrid(scales, ratios)
    scales, ratios = scales.flatten(), ratios.flatten()
    # scales = [2 4 8 2 4 8 2 4 8]
    # ratios = [0.5 0.5 0.5 1.  1.  1.  2.  2.  2. ]

    # 生成锚框尺寸
    scale_X = scales / np.sqrt(ratios)
    scale_Y = scales * np.sqrt(ratios)
    # scale_X = [ 2.82842712  5.65685425 11.3137085   2.          4.          8.
    #   1.41421356  2.82842712  5.65685425]
    # scale_Y = [ 1.41421356  2.82842712  5.65685425  2.          4.          8.
    #   2.82842712  5.65685425 11.3137085 ]
    # 对应位置两两组合得到9种Anchor框，且对称

    # 生成锚框对应原图像的坐标点
    shift_X = np.arange(0, size_X) * rpn_stride
    shift_Y = np.arange(0, size_Y) * rpn_stride
    # shift_X = [  0   8  16  24  32  40  48  56  64  72  80  88  96 104 112 120]
    # shift_Y = [  0   8  16  24  32  40  48  56  64  72  80  88  96 104 112 120]
    shift_X, shift_Y = np.meshgrid(shift_X, shift_Y)
    # shift_X 与 shift_Y 两两组合，得到锚点

    # 生成锚框中心点坐标
    center_X, anchor_X = np.meshgrid(shift_X, scale_X)
    center_Y, anchor_Y = np.meshgrid(shift_Y, scale_Y)
    # 以上四个参数shape = (9, 256)
    # scale.shape * ratios.shape = 3 * 3 = 9 表示每个锚点上有9种锚框
    # size_X * size_Y = 16 * 16 = 256 表示一共取了256个锚点，这也是 Feature Map中的元素个数
    # (center_X[i, j], center_Y[i, j], anchor_X[i, j], anchor[i, j]) 表示第i个锚点的第j个锚框的
    # (中心坐标X, 中心坐标y, 框宽, 框高)

    # 组合9*256个锚框的*中心点*坐标，得到坐标序列
    anchor_center = np.stack([center_Y, center_X], axis=2).reshape(-1, 2)   # anchor_center.shape=(9*256, 2)=(2304, 2)
    # 组合9*256个锚框的尺寸，得到尺寸序列
    anchor_size = np.stack([anchor_Y, anchor_X], axis=2).reshape(-1, 2)     # anchor_size.shape = (2304, 2)

    # 组合锚框序列，得到左上顶点坐标和右下顶点坐标
    boxes = np.concatenate([anchor_center-0.5*anchor_size, anchor_center+0.5*anchor_size], axis=1)
    # anchor_center-0.5*anchor_size: 左上顶点坐标（包含x, y）
    # boxes.shape = (Scales.shape * Ratios.shape * Size_X * Size_Y, 2)
    return boxes


if __name__ == '__main__':
    anchors = anchor_generator(Size_X, Size_Y, rpn_Stride, Scales, Ratios)

    # ————————
    # 可视化锚框
    plt.figure(figsize=(10, 10))
    img = np.ones((128, 128, 3))    # 128 = 16 * 8
    plt.imshow(img)
    Axs = plt.gca()                 # get current Axs

    img_area = 128*128
    anchor_area = 0
    for i in range(anchors.shape[0]):
        box = anchors[i]
        rec = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], edgecolor="r", facecolor="none")
        anchor_area += (box[2]-box[0])*(box[3]-box[1])
        Axs.add_patch(rec)
    print(anchor_area/img_area)
    plt.show()
    # 可视化锚框
    # ————————


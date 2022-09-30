# -*- coding: utf-8 -*-
# @Time     : 2022/9/29 20:36
# @Author   : Dyishere
# @File     : Faster_RCNN.py
"""
功能描述：

"""
import sys
sys.path.append("..")
import model
from layer.conv2d import *
from layer.pool import *
from layer.linear import *
from layer.activation import *
from layer.batchNormalization import *
import time


class Faster_RCNN(model):
    def __init__(self):
        super().__init__()

        # ————————————————————————————————————————————————————————————————————
        # ResNet
        self.conv1 = Conv2d(in_C=3, out_C=64, K_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(in_dim=64, eps=1e-05, momentum=0.1, affine=True, axis=1)
        self.relu1 = Relu()
        self.maxpool1 = MaxPool2d(K_size=3, stride=2, padding=1)

        # -Res layer 1
        # --Bottleneck 0
        # --Bottleneck 1
        # --Bottleneck 2

        # -Res layer 2
        # --Bottleneck 0
        # --Bottleneck 1
        # --Bottleneck 2
        # --Bottleneck 3

        # -Res layer 3
        # --Bottleneck 0
        # --Bottleneck 1
        # --Bottleneck 2
        # --Bottleneck 3
        # --Bottleneck 4
        # --Bottleneck 5

        # -Res layer 4
        # --Bottleneck 0
        # --Bottleneck 1
        # --Bottleneck 2
        # ————————————————————————————————————————————————————————————————————

        # ————————————————————————————————————————————————————————————————————
        # FPN
        # -lateral_convs
        # -fpn_convs
        # ————————————————————————————————————————————————————————————————————

        # ————————————————————————————————————————————————————————————————————
        # RPN Head
        # ————————————————————————————————————————————————————————————————————

        # ————————————————————————————————————————————————————————————————————
        # ROI Head
        # ————————————————————————————————————————————————————————————————————

    def forward(self, x):
        # x.shape: batch_size, in_channels, in_height, in_weight
        pass

    def backpropagation(self, d_out):
        pass


# FasterRCNN(
#   (backbone): ResNet(
#     (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#     (layer1): ResLayer(
#       (0): Bottleneck(
#         (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer2): ResLayer(
#       (0): Bottleneck(
#         (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (3): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer3): ResLayer(
#       (0): Bottleneck(
#         (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (3): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (4): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (5): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer4): ResLayer(
#       (0): Bottleneck(
#         (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#   )
#   (neck): FPN(
#     (lateral_convs): ModuleList(
#       (0): ConvModule(
#         (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#       )
#       (1): ConvModule(
#         (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
#       )
#       (2): ConvModule(
#         (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
#       )
#       (3): ConvModule(
#         (conv): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
#       )
#     )
#     (fpn_convs): ModuleList(
#       (0): ConvModule(
#         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (1): ConvModule(
#         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (2): ConvModule(
#         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (3): ConvModule(
#         (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#     )
#   )
#   (rpn_head): RPNHead(
#     (loss_cls): CrossEntropyLoss()
#     (loss_bbox): L1Loss()
#     (rpn_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (rpn_cls): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
#     (rpn_reg): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (roi_head): StandardRoIHead(
#     (bbox_roi_extractor): SingleRoIExtractor(
#       (roi_layers): ModuleList(
#         (0): RoIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=0, pool_mode=avg, aligned=True, use_torchvision=False)
#         (1): RoIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=0, pool_mode=avg, aligned=True, use_torchvision=False)
#         (2): RoIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=0, pool_mode=avg, aligned=True, use_torchvision=False)
#         (3): RoIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, pool_mode=avg, aligned=True, use_torchvision=False)
#       )
#     )
#     (bbox_head): Shared2FCBBoxHead(
#       (loss_cls): CrossEntropyLoss()
#       (loss_bbox): L1Loss()
#       (fc_cls): Linear(in_features=1024, out_features=11, bias=True)
#       (fc_reg): Linear(in_features=1024, out_features=40, bias=True)
#       (shared_convs): ModuleList()
#       (shared_fcs): ModuleList(
#         (0): Linear(in_features=12544, out_features=1024, bias=True)
#         (1): Linear(in_features=1024, out_features=1024, bias=True)
#       )
#       (cls_convs): ModuleList()
#       (cls_fcs): ModuleList()
#       (reg_convs): ModuleList()
#       (reg_fcs): ModuleList()
#       (relu): ReLU(inplace=True)
#     )
#   )
# )
# loading annotations into memory...


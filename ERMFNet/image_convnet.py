import torch, os
from torch.optim import *
from torch.autograd import *
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from GLI_CAM import GLIBlock


import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalGate(nn.Module):
    """
    修正版：真正的方向感知门控

    核心思路：使用方向条纹卷积（Directional Strip Convolution）
    捕获真正的方向性特征
    """

    def __init__(self, channels, ratio=16):
        super(DirectionalGate, self).__init__()

        mid_channels = max(channels // ratio, 16)

        # 方向1：水平条纹卷积 (1×k)
        self.horizontal_conv = nn.Conv2d(channels, mid_channels,
                                         kernel_size=(1, 7),
                                         padding=(0, 3),
                                         bias=False)

        # 方向2：垂直条纹卷积 (k×1)
        self.vertical_conv = nn.Conv2d(channels, mid_channels,
                                       kernel_size=(7, 1),
                                       padding=(3, 0),
                                       bias=False)

        # 方向3：主对角线卷积
        self.diag1_conv = nn.Conv2d(channels, mid_channels,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)

        # 方向4：副对角线卷积
        self.diag2_conv = nn.Conv2d(channels, mid_channels,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)

        # 融合4个方向的特征
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 4, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 4个方向的特征提取
        h_feat = self.horizontal_conv(x)  # 水平方向
        v_feat = self.vertical_conv(x)  # 垂直方向
        d1_feat = self.diag1_conv(x)  # 对角线1
        d2_feat = self.diag2_conv(x)  # 对角线2

        # 拼接4个方向
        dir_feats = torch.cat([h_feat, v_feat, d1_feat, d2_feat], dim=1)

        # 学习方向权重
        dir_weight = self.fusion(dir_feats)

        # 应用门控
        out = x * dir_weight

        return out


class SimpleDirectionalGate(nn.Module):
    """
    简化版：只用水平和垂直两个主要方向
    推荐用这个 - 更轻量、更清晰
    """

    def __init__(self, channels, ratio=16):
        super(SimpleDirectionalGate, self).__init__()

        mid_channels = max(channels // ratio, 16)

        # 水平方向：1×7卷积
        self.h_conv = nn.Conv2d(channels, channels,
                                kernel_size=(1, 7),
                                padding=(0, 3),
                                groups=channels,  # 深度卷积，更轻量
                                bias=False)

        # 垂直方向：7×1卷积
        self.v_conv = nn.Conv2d(channels, channels,
                                kernel_size=(7, 1),
                                padding=(3, 0),
                                groups=channels,
                                bias=False)

        # 全局上下文
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 方向权重学习
        self.fc = nn.Sequential(
            nn.Linear(channels * 3, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 提取方向特征
        h_feat = self.global_pool(self.h_conv(x)).view(b, c)  # 水平
        v_feat = self.global_pool(self.v_conv(x)).view(b, c)  # 垂直
        g_feat = self.global_pool(x).view(b, c)  # 全局

        # 拼接特征
        feat = torch.cat([h_feat, v_feat, g_feat], dim=1)

        # 学习通道权重
        weight = self.fc(feat).view(b, c, 1, 1)

        # 应用门控
        out = x * weight

        return out


class ImageConvNet(nn.Module):
    """使用SimpleDirectionalGate的网络 - 推荐版本"""

    def __init__(self):
        super(ImageConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

        # Block 1: 64 channels
        self.cnn1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bat10 = nn.BatchNorm2d(64)
        self.bat11 = nn.BatchNorm2d(64)
        self.dgate1 = SimpleDirectionalGate(64, ratio=16)
        self.dgate2 = SimpleDirectionalGate(64, ratio=16)

        # Block 2: 128 channels
        self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bat20 = nn.BatchNorm2d(128)
        self.bat21 = nn.BatchNorm2d(128)
        self.dgate3 = SimpleDirectionalGate(128, ratio=16)
        self.dgate4 = SimpleDirectionalGate(128, ratio=16)

        # Block 3: 256 channels
        self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bat30 = nn.BatchNorm2d(256)
        self.bat31 = nn.BatchNorm2d(256)
        self.dgate5 = SimpleDirectionalGate(256, ratio=16)
        self.dgate6 = SimpleDirectionalGate(256, ratio=16)

        # Block 4: 512 channels
        self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bat40 = nn.BatchNorm2d(512)
        self.bat41 = nn.BatchNorm2d(512)
        self.dgate7 = SimpleDirectionalGate(512, ratio=16)
        self.dgate8 = SimpleDirectionalGate(512, ratio=16)

        # ========== 新增部分 ==========
        # 全局平均池化，将 [B, 512, H, W] -> [B, 512, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层，将 512 维降到 128 维
        self.fc = nn.Linear(512, 128)
        # ==============================

    def forward(self, inp):
        # Block 1
        c = F.relu(self.dgate1(self.bat10(self.cnn1(inp))))
        c = F.relu(self.dgate2(self.bat11(self.cnn2(c))))
        c = self.pool(c)

        # Block 2
        c = F.relu(self.dgate3(self.bat20(self.cnn3(c))))
        c = F.relu(self.dgate4(self.bat21(self.cnn4(c))))
        c = self.pool(c)

        # Block 3
        c = F.relu(self.dgate5(self.bat30(self.cnn5(c))))
        c = F.relu(self.dgate6(self.bat31(self.cnn6(c))))
        c = self.pool(c)

        # Block 4
        c = F.relu(self.dgate7(self.bat40(self.cnn7(c))))
        c = F.relu(self.dgate8(self.bat41(self.cnn8(c))))

        # ========== 新增部分 ==========
        # 全局池化: [B, 512, H, W] -> [B, 512, 1, 1]
        c = self.global_pool(c)
        # 展平: [B, 512, 1, 1] -> [B, 512]
        c = c.view(c.size(0), -1)
        # 降维: [B, 512] -> [B, 128]
        c = self.fc(c)
        # ==============================

        return c  # 现在返回 [B, 128]

    #def loss(self, output):
    #    return (output.mean()) ** 2

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------
#  DirectionalConvBlock 类 (*** 已移除方向门控 ***)
#  现在退化为一个标准的 CBR 块 (Conv-BN-ReLU)
# ----------------------------------------------------
class DirectionalConvBlock(nn.Module):
    """
    消融实验版本：
    已移除方向门控机制 (h_conv, v_conv, fc, Gating)，
    使其退化为一个标准的 CBR 块 (Conv-BN-ReLU)。
    """

    def __init__(self, in_channels, out_channels, stride=1, ratio=16):
        super(DirectionalConvBlock, self).__init__()
        self.stride = stride
        # mid_channels_for_gate = max(out_channels // ratio, 16) # 已移除

        # 1. 核心卷积：负责通道变换和下采样
        self.conv_main = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. 方向性特征提取 (用于门控) - [已移除]
        # self.h_conv = ...
        # self.v_conv = ...

        # 3. 全局上下文和权重学习 - [已移除]
        # self.global_pool = ...
        # self.fc = ...

    def forward(self, x):
        # 1. 首先通过主卷积层，进行通道变换和下采样
        # 这一步得到的是经过 CBR 之后的特征图
        x_processed = self.relu(self.bn(self.conv_main(x)))

        # 2. 从 x_processed 提取方向特征并计算权重 - [已移除]
        # ...

        # 3. 应用方向门控权重到 x_processed - [已移除]
        # out = x_processed * weight

        # 直接返回 CBR 的结果
        return x_processed


# ----------------------------------------------------
#  ImageConvNet 类 (*** 保持不变 ***)
#  该类的架构和调用方式完全不变
# ----------------------------------------------------
class ImageConvNet(nn.Module):
    """
    使用 "Stem + 4 Stages" 架构
    每个 Stage 包含一个 DirectionalConvBlock (现在是CBR块)
    """

    def __init__(self):
        super(ImageConvNet, self).__init__()

        # 1. Stem (初始卷积层)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 保持 3 通道输入
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )  # 输出: [B, 64, H/4, W/4]

        # 2. Stages (方向模块组)
        # 这里的调用方式完全不变
        self.blocks = nn.ModuleList([
            DirectionalConvBlock(64, 128, stride=2),
            DirectionalConvBlock(128, 256, stride=2),
            DirectionalConvBlock(256, 512, stride=2),
            DirectionalConvBlock(512, 512, stride=1)
        ])

        # 3. Classifier (分类头)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        # 1. 通过 Stem
        x = self.initial_conv(x)

        # 2. 依次通过所有 Stages
        for block in self.blocks:
            x = block(x)

        # 3. 通过分类头
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)

        return x'''

if __name__ == "__main__":
	model = ImageConvNet().cuda()
	print("Model loaded.")
	image = Variable(torch.rand(2, 3, 224, 224)).cuda()
	print("Image loaded.")

	# Run a feedforward and check shape
	c = model(image)
	print(image.shape)
	print(c.shape)

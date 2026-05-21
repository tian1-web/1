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



class FrequencyBandAttention(nn.Module):
    """
    频带注意力 - 针对频谱图的频率特性设计
    """

    def __init__(self, channels, num_bands=8):
        super(FrequencyBandAttention, self).__init__()
        self.num_bands = num_bands

        # 每个 band 输出一个权重
        self.band_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # [B, C, 1, 1]
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_bands, 1),  # [B, num_bands, 1, 1]
            nn.Sigmoid()
        )

        # 频带全局重加权参数
        self.band_reweight = nn.Parameter(torch.ones(1, 1, num_bands, 1))

    def forward(self, x):
        b, c, h, w = x.shape

        # 将频率维度分成 num_bands 份
        bands = torch.chunk(x, self.num_bands, dim=2)  # list of [B, C, H/num_bands, W]

        # 每个 band 的注意力权重
        band_weights = self.band_attention(x)  # [B, num_bands, 1, 1]

        weighted_bands = []
        for i, band in enumerate(bands):
            weight = band_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            weighted_bands.append(band * weight)

        output = torch.cat(weighted_bands, dim=2)  # [B, C, H, W]

        # 应用全局重加权
        return output * self.band_reweight.mean(dim=2, keepdim=True)


class TemporalModelingBlock(nn.Module):
    """
    时序建模模块 - 针对频谱图的时间序列特性
    """

    def __init__(self, channels, kernel_size=5):
        super(TemporalModelingBlock, self).__init__()

        # 时间建模 (频谱随时间变化)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                      padding=(0, kernel_size // 2), groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 频率建模 (捕捉频率间依赖)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                      padding=(kernel_size // 2, 0), groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        temporal_feat = self.temporal_conv(x)  # 时间维度建模
        freq_feat = self.freq_conv(x)          # 频率维度建模

        gate = self.gate(torch.cat([temporal_feat, freq_feat], dim=1))
        return gate * temporal_feat + (1 - gate) * freq_feat


class SpectrogramOptimizedBlock(nn.Module):
    """
    频谱图专用模块 - 结合频带注意力和时序建模
    """

    def __init__(self, channels):
        super(SpectrogramOptimizedBlock, self).__init__()
        self.freq_attention = FrequencyBandAttention(channels)
        self.temporal_modeling = TemporalModelingBlock(channels)

        # 残差连接
        self.conv_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )

        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # 频带注意力
        x = self.freq_attention(x)

        # 时序建模
        x = self.temporal_modeling(x)

        # 残差连接
        residual = self.conv_residual(identity)
        return self.final_activation(x + residual)


class AudioConvNet(nn.Module):
    """
    音频频谱图特征提取网络
    """

    def __init__(self):
        super(AudioConvNet, self).__init__()

        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 频谱图专用块
        self.blocks = nn.ModuleList([
            self._make_spectrogram_block(64, 128, 2),
            self._make_spectrogram_block(128, 256, 2),
            self._make_spectrogram_block(256, 512, 2),
            self._make_spectrogram_block(512, 512, 1)
        ])

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc = nn.Linear(512, 128)

    def _make_spectrogram_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SpectrogramOptimizedBlock(out_channels)
        )

    def forward(self, x):
        x = self.initial_conv(x)

        for block in self.blocks:
            x = block(x)

        pooled = self.global_pool(x)
        flattened = pooled.view(pooled.size(0), -1)
        output = self.fc(flattened)

        return output

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------
#  FrequencyBandAttention 类 (*** 已按要求删除 ***)
# ----------------------------------------------------
# class FrequencyBandAttention(nn.Module):
#     ... (代码已移除)


class TemporalModelingBlock(nn.Module):
    """
    时序建模模块 - 针对频谱图的时间序列特性
    (保持不变)
    """

    def __init__(self, channels, kernel_size=5):
        super(TemporalModelingBlock, self).__init__()

        # 时间建模 (频谱随时间变化)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                      padding=(0, kernel_size // 2), groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 频率建模 (捕捉频率间依赖)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                      padding=(kernel_size // 2, 0), groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        temporal_feat = self.temporal_conv(x)  # 时间维度建模
        freq_feat = self.freq_conv(x)  # 频率维度建模

        gate = self.gate(torch.cat([temporal_feat, freq_feat], dim=1))
        return gate * temporal_feat + (1 - gate) * freq_feat


class SpectrogramOptimizedBlock(nn.Module):
    """
    频谱图专用模块 - (*** 已移除频带注意力 ***)
    """

    def __init__(self, channels):
        super(SpectrogramOptimizedBlock, self).__init__()

        # self.freq_attention = FrequencyBandAttention(channels) # <--- 已移除

        self.temporal_modeling = TemporalModelingBlock(channels)

        # 残差连接
        self.conv_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )

        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # 频带注意力 - [已移除]
        # x = self.freq_attention(x) 

        # 时序建模
        # 注意：现在 x 直接进入时序建模
        x = self.temporal_modeling(x)

        # 残差连接
        residual = self.conv_residual(identity)
        return self.final_activation(x + residual)


class AudioConvNet(nn.Module):
    """
    音频频谱图特征提取网络
    (保持不变)
    """

    def __init__(self):
        super(AudioConvNet, self).__init__()

        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 频谱图专用块
        self.blocks = nn.ModuleList([
            self._make_spectrogram_block(64, 128, 2),
            self._make_spectrogram_block(128, 256, 2),
            self._make_spectrogram_block(256, 512, 2),
            self._make_spectrogram_block(512, 512, 1)
        ])

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc = nn.Linear(512, 128)

    def _make_spectrogram_block(self, in_channels, out_channels, stride):
        # 此函数保持不变，它现在会调用修改后的 SpectrogramOptimizedBlock
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SpectrogramOptimizedBlock(out_channels)
        )

    def forward(self, x):
        x = self.initial_conv(x)

        for block in self.blocks:
            x = block(x)

        pooled = self.global_pool(x)
        flattened = pooled.view(pooled.size(0), -1)
        output = self.fc(flattened)

        return output'''



'''import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyBandAttention(nn.Module):
    """
    频带注意力 - (保持不变)
    """
    def __init__(self, channels, num_bands=8):
        super(FrequencyBandAttention, self).__init__()
        self.num_bands = num_bands
        self.band_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_bands, 1),
            nn.Sigmoid()
        )
        self.band_reweight = nn.Parameter(torch.ones(1, 1, num_bands, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        bands = torch.chunk(x, self.num_bands, dim=2)
        band_weights = self.band_attention(x)
        weighted_bands = []
        for i, band in enumerate(bands):
            weight = band_weights[:, i:i+1, :, :]
            weighted_bands.append(band * weight)
        output = torch.cat(weighted_bands, dim=2)
        return output * self.band_reweight.mean(dim=2, keepdim=True)


# ----------------------------------------------------
#  SpectrogramOptimizedBlock 类 (*** 这是您需要的修改 ***)
#  CBR 块被 "移动" 到了这个类里面
# ----------------------------------------------------
class SpectrogramOptimizedBlock(nn.Module):
    """
    频谱图专用模块 - (*** 已重构 ***)
    集成了 CBR (Conv-BN-ReLU) 和频带注意力。
    (已移除时序建模和残差连接)
    """
    def __init__(self, in_channels, out_channels, stride=1, num_bands=8):
        super(SpectrogramOptimizedBlock, self).__init__()

        # 1. 核心卷积：负责通道变换和下采样 (从 _make_... 函数移动到这里)
        self.conv_main = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=3, stride=stride, 
                                   padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. 频带注意力 (现在它在 CBR 之后工作)
        self.freq_attention = FrequencyBandAttention(out_channels, num_bands=num_bands)

        # 3. 最终激活 (来自您之前的版本)
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. 首先通过主 CBR 块 (完成通道变换和下采样)
        x = self.relu(self.bn(self.conv_main(x)))

        # 2. 然后通过频带注意力
        x = self.freq_attention(x)

        # 3. 最终激活
        return self.final_activation(x)


# ----------------------------------------------------
#  AudioConvNet 类 (*** 已修改 ***)
# ----------------------------------------------------
class AudioConvNet(nn.Module):
    """
    音频频谱图特征提取网络
    (现在使用重构后的 SpectrogramOptimizedBlock)
    """

    def __init__(self):
        super(AudioConvNet, self).__init__()

        # 1. Stem (初始卷积层)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Stages (频谱图专用块)
        # 现在直接实例化 SpectrogramOptimizedBlock
        self.blocks = nn.ModuleList([
            SpectrogramOptimizedBlock(64, 128, stride=2),
            SpectrogramOptimizedBlock(128, 256, stride=2),
            SpectrogramOptimizedBlock(256, 512, stride=2),
            SpectrogramOptimizedBlock(512, 512, stride=1)
        ])

        # 3. Classifier (分类头)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 128)

    # _make_spectrogram_block 不再需要了
    # def _make_spectrogram_block(self, in_channels, out_channels, stride):
    #     ...

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        pooled = self.global_pool(x)
        flattened = pooled.view(pooled.size(0), -1)
        output = self.fc(flattened)
        return output'''


if __name__ == "__main__":

	model = AudioConvNet().cuda()
	print("Model loaded.")
	image = Variable(torch.rand(2, 1, 257, 200)).cuda()
	print("Image loaded.")

	# Run a feedforward and check shape
	c = model(image)
	print(image.shape)
	print(c.shape)

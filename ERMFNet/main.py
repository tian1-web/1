from image_convnet import *
from audio_convnet import *
import shutil
import time
import argparse
from torch.optim import *
from torchvision.transforms import *
import warnings
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import json
from utils.mydata_xu import *
import math
from tqdm import tqdm
from prettytable import PrettyTable
from thop import profile, clever_format
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os


# ==================== 通道注意力模块 ====================
class FC_Block(nn.Module):
    """通道注意力模块"""

    def __init__(self, inplanes, planes):
        super(FC_Block, self).__init__()
        self.fc1 = nn.Linear(inplanes, planes)
        self.fc2 = nn.Linear(planes, inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.view(x.size(0), -1, 1, 1)


# ==================== 混合MSFRF-DST融合模块 ====================
class HybridMSFRF_DST(nn.Module):
    """
    混合融合策略：
    - 主路径：MSFRF多尺度特征融合（保证性能）
    - 辅助路径：DST证据理论（提供不确定性估计和额外监督）

    优势：
    1. 保留MSFRF的强大特征融合能力
    2. 引入DST的不确定性量化能力
    3. 通过辅助监督提升模型鲁棒性
    4. 提供可解释的置信度估计
    """

    def __init__(self, channels=128, num_classes=3, r=4, num_heads=4):
        super(HybridMSFRF_DST, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        inter_channels = int(channels // r)

        # ========== 主路径：MSFRF组件 ==========
        # 图像分支的局部注意力
        self.local_att_img = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 音频分支的局部注意力
        self.local_att_aud = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 1D卷积（时序建模）
        self.conv_img = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_aud = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.bn_img = nn.BatchNorm2d(channels)
        self.bn_aud = nn.BatchNorm2d(channels)

        # 通道注意力
        self.channel_att_img = FC_Block(channels, 16)
        self.channel_att_aud = FC_Block(channels, 16)

        # 交叉注意力（跨模态交互）
        self.cross_att = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        # 门控机制
        self.gate_img = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate_aud = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # ========== 辅助路径：轻量级DST组件 ==========
        # 证据生成网络
        self.img_evidence_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels // 4, num_classes),
            nn.Softplus()  # 确保证据为正
        )

        self.aud_evidence_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels // 4, num_classes),
            nn.Softplus()
        )

        # 可靠性评估网络（评估每个模态的质量）
        self.reliability_net_img = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, 1),
            nn.Sigmoid()  # 输出0-1之间的可靠性分数
        )

        self.reliability_net_aud = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, 1),
            nn.Sigmoid()
        )

        # 自适应融合权重（基于不确定性）
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(4, 16),  # 输入：两个模态的不确定性 + 冲突度 + 可靠性差异
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)  # 输出两个模态的融合权重
        )

        self.sigmoid = nn.Sigmoid()

    # ========== DST核心函数 ==========
    def evidence_to_dirichlet(self, evidence):
        """证据转换为Dirichlet参数：alpha = evidence + 1"""
        return evidence + 1.0

    def dirichlet_to_belief_uncertainty(self, alpha):
        """
        从Dirichlet参数计算信念和不确定性
        belief: 对每个类别的信念程度
        uncertainty: 总体不确定性
        """
        S = alpha.sum(dim=1, keepdim=True)
        belief = (alpha - 1.0) / S
        uncertainty = self.num_classes / S
        return belief, uncertainty

    def discount_evidence(self, evidence, reliability):
        """
        证据折扣机制：根据可靠性调整证据强度
        可靠性越低，证据被折扣得越多
        """
        return evidence * (reliability ** 2)

    def compute_conflict(self, alpha1, alpha2, b1, b2):
        """
        计算两个证据源之间的冲突度
        使用余弦相似度来衡量信念的一致性
        """
        # 归一化信念向量
        b1_norm = F.normalize(b1, p=2, dim=1)
        b2_norm = F.normalize(b2, p=2, dim=1)

        # 余弦相似度
        cosine_sim = (b1_norm * b2_norm).sum(dim=1, keepdim=True)

        # 冲突度 = 1 - 相似度
        conflict = (1.0 - cosine_sim).clamp(min=1e-7, max=0.99)

        return conflict

    def dempster_combination(self, alpha1, alpha2, conflict):
        """
        Dempster组合规则：融合两个证据源
        考虑冲突度进行归一化
        """
        # 计算信念和不确定性
        b1, u1 = self.dirichlet_to_belief_uncertainty(alpha1)
        b2, u2 = self.dirichlet_to_belief_uncertainty(alpha2)

        # Dempster组合
        combined_belief = (b1 * b2 + b1 * u2 + b2 * u1) / (1.0 - conflict + 1e-7)
        combined_uncertainty = (u1 * u2) / (1.0 - conflict + 1e-7)

        # 转回Dirichlet参数
        S = self.num_classes / (combined_uncertainty + 1e-7)
        combined_alpha = combined_belief * S + 1.0

        return combined_alpha

    def forward(self, img_feat, aud_feat):
        """
        前向传播

        Args:
            img_feat: [B, 128, 1, 1] 图像特征
            aud_feat: [B, 128, 1, 1] 音频特征

        Returns:
            fused_feat: [B, 128, 1, 1] 融合后的特征（主输出）
            fusion_info: dict 包含DST相关信息（辅助输出）
        """
        B = img_feat.size(0)

        # ==================== 主路径：MSFRF特征增强 ====================
        # 阶段1：特征交叉增强
        xa_img = img_feat + aud_feat  # 图像特征 + 音频残差
        xa_aud = aud_feat + img_feat  # 音频特征 + 图像残差

        # 阶段2：局部注意力
        img_local = self.local_att_img(xa_img)
        aud_local = self.local_att_aud(xa_aud)

        # 阶段3：1D卷积（时序建模）
        img_conv = self.conv_img(xa_img.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        img_conv = self.bn_img(img_conv)
        aud_conv = self.conv_aud(xa_aud.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        aud_conv = self.bn_aud(aud_conv)

        # 阶段4：交叉注意力（跨模态交互）
        img_flat = img_feat.view(B, self.channels, -1).transpose(1, 2)  # [B, HW, C]
        aud_flat = aud_feat.view(B, self.channels, -1).transpose(1, 2)

        cross_img, _ = self.cross_att(img_flat, aud_flat, aud_flat)  # Q=img, K/V=aud
        cross_img = cross_img.transpose(1, 2).view(B, self.channels, 1, 1)

        cross_aud, _ = self.cross_att(aud_flat, img_flat, img_flat)  # Q=aud, K/V=img
        cross_aud = cross_aud.transpose(1, 2).view(B, self.channels, 1, 1)

        # 阶段5：门控融合
        gate_img = self.gate_img(torch.cat([img_feat, aud_feat], dim=1))
        gate_aud = self.gate_aud(torch.cat([aud_feat, img_feat], dim=1))

        # 阶段6：多尺度特征融合
        fusion_img = img_local + img_conv + cross_img
        fusion_aud = aud_local + aud_conv + cross_aud

        # 阶段7：通道注意力
        wei_img = self.sigmoid(self.channel_att_img(fusion_img))
        wei_aud = self.sigmoid(self.channel_att_aud(fusion_aud))

        # 阶段8：最终增强特征
        img_enhanced = img_feat * gate_img + aud_feat * (1 - gate_img) + wei_img * fusion_img
        aud_enhanced = aud_feat * gate_aud + img_feat * (1 - gate_aud) + wei_aud * fusion_aud

        # ==================== 辅助路径：DST证据融合 ====================
        # 全局特征提取
        img_global = F.adaptive_avg_pool2d(img_enhanced, 1).view(B, -1)
        aud_global = F.adaptive_avg_pool2d(aud_enhanced, 1).view(B, -1)

        # 证据生成
        evidence_img = self.img_evidence_net(img_enhanced)  # [B, num_classes]
        evidence_aud = self.aud_evidence_net(aud_enhanced)

        # 可靠性评估
        reliability_img = self.reliability_net_img(img_global)  # [B, 1]
        reliability_aud = self.reliability_net_aud(aud_global)

        # 证据折扣（根据可靠性调整）
        evidence_img_discounted = self.discount_evidence(evidence_img, reliability_img)
        evidence_aud_discounted = self.discount_evidence(evidence_aud, reliability_aud)

        # 转为Dirichlet参数
        alpha_img = self.evidence_to_dirichlet(evidence_img_discounted)
        alpha_aud = self.evidence_to_dirichlet(evidence_aud_discounted)

        # 计算信念和不确定性
        belief_img, uncertainty_img = self.dirichlet_to_belief_uncertainty(alpha_img)
        belief_aud, uncertainty_aud = self.dirichlet_to_belief_uncertainty(alpha_aud)

        # 计算冲突度
        conflict = self.compute_conflict(alpha_img, alpha_aud, belief_img, belief_aud)

        # Dempster组合
        alpha_fused = self.dempster_combination(alpha_img, alpha_aud, conflict)

        # 最终DST输出
        belief_fused, uncertainty_fused = self.dirichlet_to_belief_uncertainty(alpha_fused)
        S = alpha_fused.sum(dim=1, keepdim=True)
        prob = alpha_fused / S  # 预测概率

        # ==================== 自适应融合策略 ====================
        # 构建决策输入：不确定性 + 冲突度 + 可靠性差异
        reliability_diff = torch.abs(reliability_img - reliability_aud)
        decision_input = torch.cat([
            uncertainty_img,
            uncertainty_aud,
            conflict,
            reliability_diff
        ], dim=1)  # [B, 4]

        # 计算自适应权重
        adaptive_weights = self.adaptive_weight_net(decision_input)  # [B, 2]
        w_img = adaptive_weights[:, 0:1].view(B, 1, 1, 1)
        w_aud = adaptive_weights[:, 1:2].view(B, 1, 1, 1)

        # ==================== 主输出：融合特征 ====================
        # 策略：使用自适应权重融合增强后的特征
        fused_feat = w_img * img_enhanced + w_aud * aud_enhanced

        # ==================== 返回结果 ====================
        fusion_info = {
            # DST相关
            'alpha': alpha_fused,
            'prob': prob,
            'belief': belief_fused,
            'uncertainty': uncertainty_fused,
            'conflict': conflict,

            # 单模态信息
            'evidence_img': evidence_img,
            'evidence_aud': evidence_aud,
            'evidence_img_discounted': evidence_img_discounted,
            'evidence_aud_discounted': evidence_aud_discounted,
            'reliability_img': reliability_img,
            'reliability_aud': reliability_aud,
            'uncertainty_img': uncertainty_img,
            'uncertainty_aud': uncertainty_aud,

            # 融合权重
            'adaptive_weights': adaptive_weights,
        }

        return fused_feat, fusion_info


# ==================== 混合损失函数 ====================
class HybridLoss(nn.Module):
    """
    混合损失函数：
    1. 主损失：交叉熵（保证分类性能）
    2. 辅助损失：DST正则化（提供额外监督）
       - KL散度正则（证据正则化）
       - 冲突惩罚（鼓励模态一致性）
       - 可靠性正则（平衡两个模态）
    """

    def __init__(self, num_classes=3,
                 lambda_kl=0.05,  # KL正则权重
                 lambda_conflict=0.01,  # 冲突惩罚权重
                 lambda_reliability=0.02):  # 可靠性正则权重
        super(HybridLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl
        self.lambda_conflict = lambda_conflict
        self.lambda_reliability = lambda_reliability

    def kl_divergence_loss(self, alpha, target):
        """
        KL散度正则化：鼓励模型对正确类别产生强证据

        基本思想：
        - 对于正确类别：希望alpha值大（强证据）
        - 对于错误类别：希望alpha值接近1（弱证据）
        """
        one_hot = F.one_hot(target, self.num_classes).float()

        # 目标Dirichlet分布
        alpha_tilde = one_hot * alpha + (1 - one_hot) * 1.0

        S = alpha.sum(dim=1)
        S_tilde = alpha_tilde.sum(dim=1)

        # KL散度
        kl_loss = (
                torch.lgamma(S) - torch.lgamma(S_tilde)
                - (torch.lgamma(alpha) - torch.lgamma(alpha_tilde)).sum(dim=1)
                + ((alpha - alpha_tilde) *
                   (torch.digamma(alpha) - torch.digamma(S.unsqueeze(1)))).sum(dim=1)
        )

        return kl_loss.mean()

    def conflict_penalty(self, conflict):
        """
        冲突惩罚：鼓励两个模态产生一致的预测
        冲突度越高，惩罚越大
        """
        return conflict.mean()

    def reliability_regularization(self, reliability_img, reliability_aud):
        """
        可靠性正则化：避免模型过度依赖单一模态
        鼓励两个模态的可靠性相近
        """
        # 方差惩罚：鼓励两个可靠性值接近
        reliability_var = torch.var(torch.cat([reliability_img, reliability_aud], dim=1), dim=1)
        return reliability_var.mean()

    def forward(self, logits, fusion_info, target, epoch):
        """
        计算总损失

        Args:
            logits: [B, num_classes] 分类器输出
            fusion_info: dict DST融合信息
            target: [B] 真实标签
            epoch: int 当前epoch（用于动态权重调整）
        """
        alpha = fusion_info['alpha']
        conflict = fusion_info['conflict']
        reliability_img = fusion_info['reliability_img']
        reliability_aud = fusion_info['reliability_aud']

        # 1. 主损失：交叉熵
        ce_loss = F.cross_entropy(logits, target)

        # 2. KL散度正则
        kl_loss = self.kl_divergence_loss(alpha, target)

        # 3. 冲突惩罚
        conflict_loss = self.conflict_penalty(conflict)

        # 4. 可靠性正则
        reliability_loss = self.reliability_regularization(reliability_img, reliability_aud)

        # 动态权重调整（随训练进行逐步增加正则化强度）
        lambda_kl = min(self.lambda_kl, self.lambda_kl * epoch / 30)
        lambda_conflict = self.lambda_conflict
        lambda_reliability = min(self.lambda_reliability, self.lambda_reliability * epoch / 20)

        # 总损失
        total_loss = (
                ce_loss
                + lambda_kl * kl_loss
                + lambda_conflict * conflict_loss
                + lambda_reliability * reliability_loss
        )

        # 返回损失详情（用于日志记录）
        loss_dict = {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'kl': kl_loss.item(),
            'conflict': conflict_loss.item(),
            'reliability': reliability_loss.item(),
        }

        return total_loss, loss_dict


# ==================== 完整网络 ====================
class AVENet_Hybrid(nn.Module):
    """使用混合MSFRF-DST融合的网络"""

    def __init__(self):
        super(AVENet_Hybrid, self).__init__()

        self.relu = F.relu
        self.imgnet = ImageConvNet()
        self.audnet = AudioConvNet()

        # 使用混合融合模块
        self.fusion = HybridMSFRF_DST(channels=128, num_classes=3, r=4, num_heads=4)

        # Vision subnetwork
        self.vfc1 = nn.Linear(128, 128)
        self.vfc2 = nn.Linear(128, 128)
        self.vl2norm = nn.BatchNorm1d(128)

        # Audio subnetwork
        self.afc1 = nn.Linear(128, 128)
        self.afc2 = nn.Linear(128, 128)
        self.al2norm = nn.BatchNorm1d(128)

        # Classification
        self.fc3 = nn.Linear(128, 3)

    def forward(self, image, audio):
        n = image.size(0)

        # Image feature
        img = self.imgnet(image)
        img = self.relu(self.vfc1(img))
        img = self.vfc2(img)
        img = self.vl2norm(img).view(n, -1, 1, 1)

        # Audio feature
        aud = self.audnet(audio)
        aud = self.relu(self.afc1(aud))
        aud = self.afc2(aud)
        aud = self.al2norm(aud).view(n, -1, 1, 1)

        # 混合融合
        fused_feat, fusion_info = self.fusion(img, aud)

        # 分类
        out = fused_feat.squeeze(2).squeeze(2)
        out = self.fc3(out)

        return out, img, aud, fusion_info

    def get_image_embeddings(self, image):
        img = self.imgnet(image)
        img = self.relu(self.vfc1(img))
        img = self.vfc2(img)
        img = self.vl2norm(img)
        return img


# ==================== 工具类 ====================
class valConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1"]

        f1_list = []
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1])
            f1_list.append(F1)

        print(table)

        total_TP = 0
        total_FP = 0
        total_FN = 0

        for i in range(self.num_classes):
            total_TP += self.matrix[i, i]
            total_FP += np.sum(self.matrix[i, :]) - self.matrix[i, i]
            total_FN += np.sum(self.matrix[:, i]) - self.matrix[i, i]

        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) != 0 else 0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) != 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) != 0 else 0

        print(f"Micro-average - Precision: {micro_precision:.3f}, Recall: {micro_recall:.3f}, F1: {micro_f1:.3f}")

        return f1_list

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


class testConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1])
        print(table)

        total_TP = 0
        total_FP = 0
        total_FN = 0

        for i in range(self.num_classes):
            total_TP += self.matrix[i, i]
            total_FP += np.sum(self.matrix[i, :]) - self.matrix[i, i]
            total_FN += np.sum(self.matrix[:, i]) - self.matrix[i, i]

        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) != 0 else 0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) != 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) != 0 else 0

        print(f"Micro-average - Precision: {micro_precision:.3f}, Recall: {micro_recall:.3f}, F1: {micro_f1:.3f}")

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


class LossAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    def getacc(self):
        return (self.sum * 100) / self.count


class TestMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    def getacc(self):
        return (self.sum * 100) / self.count


# ==================== 辅助函数 ====================
def benchmark_fps(model, img_shape=(1, 3, 224, 224),
                  aud_shape=(1, 1, 257, 200),
                  use_cuda=True,
                  warmup=10,
                  repeat=100):
    """测试推理速度"""
    model.eval()
    dummy_img = torch.randn(img_shape)
    dummy_aud = torch.randn(aud_shape)
    if use_cuda:
        dummy_img = dummy_img.cuda()
        dummy_aud = dummy_aud.cuda()

    for _ in range(warmup):
        _ = model(dummy_img, dummy_aud)

    torch.cuda.synchronize() if use_cuda else None
    tic = time.perf_counter()
    for _ in range(repeat):
        _ = model(dummy_img, dummy_aud)
    torch.cuda.synchronize() if use_cuda else None
    toc = time.perf_counter()

    fps = repeat / (toc - tic)
    print(f"Inference FPS: {fps:.2f}")
    model.train()
    return fps


def calculate_params_and_flops(model, img_shape=(1, 3, 224, 224),
                               aud_shape=(1, 1, 257, 200),
                               use_cuda=True):
    """计算参数量和计算量"""
    model.eval()
    dummy_img = torch.randn(img_shape)
    dummy_aud = torch.randn(aud_shape)
    if use_cuda:
        dummy_img = dummy_img.cuda()
        dummy_aud = dummy_aud.cuda()

    flops, params = profile(model, inputs=(dummy_img, dummy_aud), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Total Parameters: {params}")
    print(f"Total FLOPs: {flops}")
    model.train()
    return params, flops


def getAVENet(use_cuda):
    """创建混合模型"""
    model = AVENet_Hybrid()
    if use_cuda:
        model = model.cuda()
    return model


# ==================== Demo函数 ====================
def demo():
    """测试模型"""
    model = AVENet_Hybrid()
    image = Variable(torch.rand(2, 3, 224, 224))
    audio = Variable(torch.rand(2, 1, 257, 200))

    out, v, a, fusion_info = model(image, audio)
    print(f"Image shape: {image.shape}, Audio shape: {audio.shape}")
    print(f"Visual feat: {v.shape}, Audio feat: {a.shape}, Output: {out.shape}")
    print(f"\nFusion info keys: {fusion_info.keys()}")

    print("\n=== DST融合信息 ===")
    print(f"图像可靠性: {fusion_info['reliability_img'][0].item():.3f}")
    print(f"音频可靠性: {fusion_info['reliability_aud'][0].item():.3f}")
    print(f"冲突度: {fusion_info['conflict'][0].item():.3f}")
    print(f"不确定性: {fusion_info['uncertainty'][0].item():.3f}")
    print(f"自适应权重: 图像={fusion_info['adaptive_weights'][0, 0].item():.3f}, "
          f"音频={fusion_info['adaptive_weights'][0, 1].item():.3f}")
    print(f"预测概率: {fusion_info['prob'][0]}")


# ==================== 主训练函数 ====================
def main(use_cuda=True, EPOCHS=200, batch_size=8, model_name="1_hybrid_model.pt"):
    """
    混合MSFRF-DST训练函数
    """
    # 模型初始化
    model = getAVENet(use_cuda)
    params_m, flops_g = calculate_params_and_flops(model, use_cuda=use_cuda)
    benchmark_fps(model, use_cuda=use_cuda)

    checkpoint_dir = r'/root/1/ERMFNet/model_save'

    # 加载预训练模型
    model_path = os.path.join(checkpoint_dir, model_name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("✅ Loading from previous checkpoint.")
    else:
        print("🆕 Training from scratch.")

    # 数据集加载
    print("\n" + "=" * 80)
    print("📊 Loading Datasets...")
    print("=" * 80)

    dataset = Mydata(
        img_speed_path=r'/root/1/ERMFNet/total/train.txt',
        img_path=r'/root/1/ERMFNet/dataset_total/train/img',
        speed_path=r'/root/1/ERMFNet/dataset_total/train/speed'
    )

    valdataset = Mydata(
        img_speed_path=r'/root/1/ERMFNet/total/val.txt',
        img_path=r'/root/1/ERMFNet/dataset_total/val/img',
        speed_path=r'/root/1/ERMFNet/dataset_total/val/speed'
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"✅ Training samples: {len(dataset)}")
    print(f"✅ Validation samples: {len(valdataset)}")

    # 损失函数和优化器
    criterion = HybridLoss(
        num_classes=3,
        lambda_kl=0.05,
        lambda_conflict=0.01,
        lambda_reliability=0.02
    )
    if use_cuda:
        criterion = criterion.cuda()
    print("\n✅ Loaded HYBRID MSFRF-DST loss function.")

    optim = SGD(model.parameters(), lr=0.25e-3, momentum=0.9, weight_decay=1e-4)
    print("✅ Optimizer loaded (SGD with momentum).")
    print("=" * 80 + "\n")

    model.train()

    # 训练循环
    try:
        best_precision = 0
        lowest_loss = 100000
        best_avgf1 = 0
        best_weightf1 = 0

        for epoch in range(EPOCHS):
            # 学习率调整
            if 50 <= epoch < 100:
                optim = SGD(model.parameters(), lr=0.25e-4, momentum=0.9, weight_decay=1e-4)
                if epoch == 50:
                    print(f"📉 Learning rate adjusted to 0.25e-4 at epoch {epoch}")
            if epoch >= 100:
                optim = SGD(model.parameters(), lr=0.25e-5, momentum=0.9, weight_decay=1e-4)
                if epoch == 100:
                    print(f"📉 Learning rate adjusted to 0.25e-5 at epoch {epoch}")

            train_losses = LossAverageMeter()
            train_acc = AccAverageMeter()

            epoch_start_time = time.time()

            # 训练阶段
            print(f"\n{'🚂 TRAINING':<20} Epoch [{epoch + 1}/{EPOCHS}]")
            print("-" * 80)

            for subepoch, (img, aud, out) in enumerate(dataloader):
                optim.zero_grad()

                out = out.squeeze(1)
                idx = (out != 3).numpy().astype(bool)
                if idx.sum() == 0:
                    continue

                img = torch.Tensor(img.numpy()[idx, :, :, :])
                aud = torch.Tensor(aud.numpy()[idx, :, :, :])
                out = torch.LongTensor(out.numpy()[idx])

                img = Variable(img)
                aud = Variable(aud)
                out = Variable(out)

                M = img.shape[0]
                if use_cuda:
                    img = img.cuda()
                    aud = aud.cuda()
                    out = out.cuda()

                # 前向传播
                o, _, _, fusion_info = model(img, aud)

                # 计算损失
                loss, loss_dict = criterion(o, fusion_info, out, epoch)

                train_losses.update(loss.item(), M)
                loss.backward()
                optim.step()

                # 计算准确率
                o = F.softmax(o, 1)
                _, ind = o.max(1)
                accuracy = (ind.data == out.data).sum() * 1.0 / M
                train_acc.update((ind.data == out.data).sum() * 1.0, M)

                if subepoch % 400 == 0:
                    print(f"  Batch [{subepoch:4d}/{len(dataloader)}] | "
                          f"Loss: {loss_dict['total']:.4f} "
                          f"(CE:{loss_dict['ce']:.4f} KL:{loss_dict['kl']:.4f} "
                          f"Conf:{loss_dict['conflict']:.4f} Rel:{loss_dict['reliability']:.4f}) | "
                          f"Acc: {accuracy * 100:5.2f}% | "
                          f"Avg Acc: {train_acc.getacc():5.2f}%")

            epoch_train_time = time.time() - epoch_start_time
            print("-" * 80)
            print(f"✅ TRAINING FINISHED | Loss: {train_losses.avg:.4f} | "
                  f"Acc: {train_acc.getacc():.2f}% | Time: {epoch_train_time:.1f}s")

            # 验证阶段
            print(f"\n{'🔍 VALIDATION':<20} Epoch [{epoch + 1}/{EPOCHS}]")
            print("-" * 80)

            val_start_time = time.time()
            val_losses = LossAverageMeter()
            val_acc = AccAverageMeter()
            labels = ['Normal', 'Aggressive', 'Drowsy']
            valconfusion = valConfusionMatrix(num_classes=3, labels=labels)

            model.eval()
            for sepoch, (img, aud, out) in enumerate(valdataloader):
                out = out.squeeze(1)
                idx = (out != 3).numpy().astype(bool)
                if idx.sum() == 0:
                    continue

                img = torch.Tensor(img.numpy()[idx, :, :, :])
                aud = torch.Tensor(aud.numpy()[idx, :, :, :])
                out = torch.LongTensor(out.numpy()[idx])

                img = Variable(img)
                aud = Variable(aud)
                out = Variable(out)

                M = img.shape[0]
                if use_cuda:
                    img = img.cuda()
                    aud = aud.cuda()
                    out = out.cuda()

                with torch.no_grad():
                    o, _, _, fusion_info = model(img, aud)
                    valloss, val_loss_dict = criterion(o, fusion_info, out, epoch)

                val_losses.update(valloss.item(), M)

                o = F.softmax(o, 1)
                _, ind = o.max(1)
                valconfusion.update(ind.to("cpu").numpy(), out.to("cpu").numpy())
                valaccuracy = (ind.data == out.data).sum() * 1.0 / M
                val_acc.update((ind.data == out.data).sum() * 1.0, M)

            model.train()

            # F1分数
            f1_scores = valconfusion.summary()
            avgf1 = sum(f1_scores) / len(f1_scores)
            weightnor = 0.399
            weightagg = 0.257
            weightdrow = 0.344
            weightf1 = (f1_scores[0] * weightnor +
                        f1_scores[1] * weightagg +
                        f1_scores[2] * weightdrow)

            val_time = time.time() - val_start_time
            print("-" * 80)
            print(f"✅ VALIDATION FINISHED | Loss: {val_losses.avg:.4f} | "
                  f"Acc: {val_acc.getacc():.2f}% | "
                  f"Avg F1: {avgf1:.4f} | Weighted F1: {weightf1:.4f} | "
                  f"Time: {val_time:.1f}s")

            # 更新最佳指标
            is_best_avgf1 = avgf1 > best_avgf1
            is_best_weightf1 = weightf1 > best_weightf1
            is_best = val_acc.getacc() > best_precision
            is_lowest_loss = val_losses.avg < lowest_loss

            best_precision = max(val_acc.getacc(), best_precision)
            lowest_loss = min(val_losses.avg, lowest_loss)
            best_avgf1 = max(avgf1, best_avgf1)
            best_weightf1 = max(weightf1, best_weightf1)

            print(f"\n{'📊 BEST METRICS SO FAR':^80}")
            print("=" * 80)
            print(f"  🏆 Best Accuracy    : {best_precision:6.2f}%")
            print(f"  📉 Lowest Loss      : {lowest_loss:6.4f}")
            print(f"  📈 Best Avg F1      : {best_avgf1:6.4f}")
            print(f"  🎯 Best Weighted F1 : {best_weightf1:6.4f}")
            print("=" * 80)

            # 保存模型
            save_path = os.path.join(checkpoint_dir, model_name)
            torch.save(model.state_dict(), save_path)

            if is_best:
                best_path = os.path.join(checkpoint_dir, '1_hybrid_best_model.pt')
                shutil.copyfile(save_path, best_path)
                print(f"🏆 NEW BEST ACCURACY MODEL saved!")

            if is_best_avgf1:
                best_avgf1_path = os.path.join(checkpoint_dir, '1_hybrid_best_avgf1.pt')
                shutil.copyfile(save_path, best_avgf1_path)
                print(f"📈 NEW BEST AVG F1 MODEL saved!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "1_hybrid_interrupted.pt"))

    finally:
        print("\n🏁 TRAINING COMPLETED")
        print(f"Best Accuracy: {best_precision:.2f}%")
        print(f"Best Avg F1: {best_avgf1:.4f}")


# ==================== 测试函数 ====================
def test(use_cuda=True, batch_size=8,
         model_name=r'/root/1/ERMFNet/model_save/hybrid_model.pt'):
    model = getAVENet(use_cuda)

    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name), strict=False)
        print("Loading from checkpoint:", model_name)
    else:
        print("Model file not found!")
        return

    testdataset = Mydata(
        img_speed_path=r'/root/1/ERMFNet/total/test.txt',
        img_path=r'/root/1/ERMFNet/dataset_total/test/img',
        speed_path=r'/root/1/ERMFNet/dataset_total/test/speed'
    )

    testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = HybridLoss(num_classes=3)
    if use_cuda:
        criterion = criterion.cuda()

    test_losses = LossAverageMeter()
    test_acc = TestMeter()
    labels = ['Normal', 'Aggressive', 'Drowsy']
    testconfusion = testConfusionMatrix(num_classes=3, labels=labels)

    model.eval()
    print("\n🧪 TESTING...")

    for sepoch, (img, aud, out) in enumerate(testdataloader):
        out = out.squeeze(1)
        idx = (out != 3).numpy().astype(bool)
        if idx.sum() == 0:
            continue

        img = torch.Tensor(img.numpy()[idx, :, :, :])
        aud = torch.Tensor(aud.numpy()[idx, :, :, :])
        out = torch.LongTensor(out.numpy()[idx])

        img = Variable(img)
        aud = Variable(aud)
        out = Variable(out)

        M = img.shape[0]
        if use_cuda:
            img = img.cuda()
            aud = aud.cuda()
            out = out.cuda()

        with torch.no_grad():
            o, _, _, fusion_info = model(img, aud)
            testloss, _ = criterion(o, fusion_info, out, epoch=0)

        test_losses.update(testloss.item(), M)

        o = F.softmax(o, 1)
        _, ind = o.max(1)
        testconfusion.update(ind.to("cpu").numpy(), out.to("cpu").numpy())
        testaccuracy = (ind.data == out.data).sum() * 1.0 / M
        test_acc.update((ind.data == out.data).sum() * 1.0, M)

        if sepoch % 100 == 0:
            print(f"Batch [{sepoch}/{len(testdataloader)}] | "
                  f"Loss: {test_losses.avg:.4f} | "
                  f"Acc: {test_acc.getacc():.2f}%")

    print("\n" + "=" * 80)
    print("TESTING FINISHED")
    print(f"Final Loss: {test_losses.avg:.4f}")
    print(f"Final Accuracy: {test_acc.getacc():.2f}%")
    print("=" * 80)

    testconfusion.summary()
    testconfusion.plot()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    cuda = True

    choices = ["demo", "main", "test"]

    parser = argparse.ArgumentParser(description="Hybrid MSFRF-DST Fusion")
    parser.add_argument('--mode', default="main", choices=choices, type=str)

    args = parser.parse_args()
    mode = args.mode

    print("=" * 80)
    print("🚀 HYBRID MSFRF-DST FUSION FOR DRIVING BEHAVIOR RECOGNITION")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print("\n📋 Model Architecture:")
    print("  ✅ Main Path: MSFRF Multi-scale Feature Fusion")
    print("  ✅ Auxiliary Path: DST Evidence Theory")
    print("  ✅ Adaptive Fusion Strategy")
    print("  ✅ Uncertainty Quantification")
    print("=" * 80)
    print()

    if mode == "demo":
        demo()
    elif mode == "main":
        main(use_cuda=cuda, batch_size=16, EPOCHS=200, model_name="1_hybrid_model.pt")
    elif mode == "test":
        test(use_cuda=cuda, batch_size=16)
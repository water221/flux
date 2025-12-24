import torch
import torch.nn as nn
import torch.nn.functional as F

def haar_wavelet_transform(x):
    """
    对输入张量执行单级离散小波变换 (Haar)。
    Args:
        x: (B, C, H, W)
    Returns:
        LL, LH, HL, HH: 四个子带，形状均为 (B, C, H/2, W/2)
    """
    b, c, h, w = x.shape
    x_pad = x
    if h % 2 != 0 or w % 2 != 0:
        x_pad = F.pad(x, (0, w % 2, 0, h % 2), mode='reflect')

    x0 = x_pad[:, :, 0::2, 0::2]  # 偶行偶列
    x1 = x_pad[:, :, 0::2, 1::2]  # 偶行奇列
    x2 = x_pad[:, :, 1::2, 0::2]  # 奇行偶列
    x3 = x_pad[:, :, 1::2, 1::2]  # 奇行奇列

    LL = (x0 + x1 + x2 + x3) / 2  # 低频近似
    LH = (x0 - x1 + x2 - x3) / 2  # 水平细节
    HL = (x0 + x1 - x2 - x3) / 2  # 垂直细节
    HH = (x0 - x1 - x2 + x3) / 2  # 对角细节

    return LL, LH, HL, HH

class WaveletLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_LL, pred_LH, pred_HL, pred_HH = haar_wavelet_transform(pred)
        target_LL, target_LH, target_HL, target_HH = haar_wavelet_transform(target)

        loss_LL = F.mse_loss(pred_LL, target_LL)
        loss_high = F.mse_loss(pred_LH, target_LH) + \
                    F.mse_loss(pred_HL, target_HL) + \
                    F.mse_loss(pred_HH, target_HH)
        
        # 加大高频部分的权重
        return loss_LL + self.alpha * loss_high
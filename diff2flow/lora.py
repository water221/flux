import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Any
import math
import torch.fft

# =============================================================================
# zw [创新点 3 辅助组件] 全局时间嵌入上下文
# 用于在 TimeAwareDualAdapter 中传递时间嵌入信息
# 使用 Context Manager 方式避免修改复杂的 UNet 递归调用结构
# =============================================================================
CURRENT_TIME_EMB = None


def getattr_recursive(obj: Any, path: str) -> Any:
    parts = path.split('.')
    for part in parts:
        if part.isnumeric():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj

target_modules: list[str] = ["to_q", "to_k", "to_v", "query", "key", "value"]

def setattr_recursive(obj: Any, path: str, value: Any) -> None:
    parts = path.split('.')
    for part in parts[:-1]:
        if part.isnumeric():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

class DataProvider:
    def __init__(self):
        self.batch = None

    def set_batch(self, batch):
        if self.batch is not None:
            if isinstance(self.batch, torch.Tensor):
                assert self.batch.shape[1:] == batch.shape[1:], "Check: shapes probably should not change during training"

        self.batch = batch

    def get_batch(self, x=None):
        assert self.batch is not None, "Error: need to set a batch first"

        if x is None or isinstance(self.batch, torch.Tensor):
            return self.batch

        # batch is a list; select the corresponding element based on x
        size = x.shape[2]
        for i in range(len(self.batch)):
            if self.batch[i].shape[2] == size:
                return self.batch[i]
            
        raise ValueError("Error: no matching batch found")

    def reset(self):
        self.batch = None

class LoraLinear(torch.nn.Module):
    def __init__(
        self,
        out_features,
        in_features,
        rank = None,
        lora_scale = 1.0,
    ):
        super().__init__()
        self.rank = rank
        self.lora_scale = lora_scale

        # original weight of the matrix
        self.W = nn.Linear(in_features, out_features, bias=False)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # b should be init wiht 0
        for p in self.B.parameters():
            p.detach().zero_()

    def forward(self, x):
        w_out = self.W(x)
        a_out = self.A(x)
        b_out = self.B(a_out)
        return w_out + b_out * self.lora_scale
    

class LoRAConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        rank: int = None,
        lora_scale: float = 1.0,
    ):
        super().__init__()

        # self.lora_scale = alpha / rank
        self.lora_scale = lora_scale

        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_normal_(self.A.weight, a=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): In shape of (B, C, H, W)
        """
        w_out = self.W(x)
        a_out = self.A(x)
        b_out = self.B(a_out)

        return w_out + b_out * self.lora_scale
    

class LoRAAdapterConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        data_provider: DataProvider,
        c_dim: int,
        rank: int = None,
        lora_scale: float = 1.0,
    ):
        super().__init__()

        # self.lora_scale = alpha / rank
        self.lora_scale = lora_scale
        self.c_dim = c_dim

        self.data_provider = data_provider

        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_normal_(self.A.weight, a=1)

        self.beta = nn.Conv2d(c_dim, rank, kernel_size=1, bias=False)
        self.gamma = nn.Conv2d(c_dim, rank, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): In shape of (B, C, H, W)
        """
        w_out = self.W(x)
        a_out = self.A(x)

        # inject conditioning into LoRA
        c = self.data_provider.get_batch(a_out)
        element_shift = self.beta(c)
        element_scale = self.gamma(c) + 1
        a_cond = a_out * element_scale + element_shift

        b_out = self.B(a_cond)

        return w_out + b_out * self.lora_scale
# =============================================================================
# zw [创新点 1 核心组件] 频域门控模块
# 理论依据：Global Filter Networks
# 利用 FFT 实现全局特征混合，捕捉长距离依赖
# =============================================================================
class SpectralGating(nn.Module):
    """
    频域门控模块：实现全局特征混合
    理论依据：Global Filter Networks
    """
    def __init__(self, dim, h=None, w=None):
        super().__init__()
        self.dim = dim
        # 使用复数权重，因为 FFT 结果是复数
        # 这里的 scale 初始化很小，保证初始状态接近恒等映射，利于训练稳定
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # 1. 2D FFT
        # rfft2 只计算一半的频率，节省空间 (针对实数输入)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 频谱滤波 (Spectral Gating)
        # 将权重扩展以匹配广播机制
        weight = torch.view_as_complex(self.complex_weight)
        weight = weight.view(1, C, 1, 1)
        
        # 频域乘法 = 空间域全局卷积
        x_fft = x_fft * weight
        
        # 3. 2D IFFT
        x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        return x


# =============================================================================
# zw [创新点 2 核心组件] 跨域交互模块
# 解决空域和频域特征割裂问题，通过 Attention Map 进行相互校准
# =============================================================================
class CrossDomainInteraction(nn.Module):
    """
    跨域交互模块：解决空域和频域特征割裂问题
    通过 Attention Map 进行双域特征的相互校准
    """
    def __init__(self, dim):
        super().__init__()
        # 降维生成门控系数，降低计算量
        self.reduce = nn.Conv2d(dim * 2, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial_f, spectral_f):
        """
        Args:
            spatial_f: 空间域特征 (B, C, H, W)
            spectral_f: 频域特征 (B, C, H, W)
        Returns:
            spatial_refined: 校准后的空间域特征
            spectral_refined: 校准后的频域特征
        """
        # 1. 拼接双域特征
        combined = torch.cat([spatial_f, spectral_f], dim=1)
        
        # 2. 生成交互门控 (Attention Map)
        gate = self.sigmoid(self.reduce(combined))
        
        # 3. 相互校准 (Rectification)
        # 频域全局信息 -> 抑制空域噪声 (gate)
        # 空域局部细节 -> 补充频域边界 (1 - gate)
        spatial_refined = spatial_f * gate
        spectral_refined = spectral_f * (1 - gate)
        
        return spatial_refined, spectral_refined

class DualDomainAdapter(nn.Module):
    """
    创新点核心：双域流形适配器
    结合了 LoRA 的低秩特性和频域的全局感知能力
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        rank: int = 16, # 建议稍微增大 rank 以容纳频域信息
        lora_scale: float = 1.0,
    ):
        super().__init__()
        self.lora_scale = lora_scale

        # 1. 冻结的原始权重 (SD2.1 Pretrained)
        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters():
            p.requires_grad_(False)

        # 2. 空间路径 (Spatial Path) - 类似于标准 LoRA
        # 捕捉高频细节 (Edges)
        self.spatial_down = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.spatial_up = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)

        # 3. 频谱路径 (Spectral Path) - 创新部分
        # 捕捉低频/全局结构 (Global Structure)
        # 使用 1x1 卷积降维，然后进 FFT，再升维
        # FIX: 必须使用 stride 以匹配空间路径的下采样
        self.spectral_down = nn.Conv2d(in_channels, rank, 1, stride, 0, bias=False)
        self.spectral_gate = SpectralGating(rank)
        self.spectral_up = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)

        # 4. 融合系数 (可学习)
        self.alpha_spatial = nn.Parameter(torch.tensor(1.0))
        self.alpha_spectral = nn.Parameter(torch.tensor(1.0))

        # 初始化
        nn.init.kaiming_uniform_(self.spatial_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.spatial_up.weight)
        nn.init.kaiming_uniform_(self.spectral_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.spectral_up.weight)

    def forward(self, x):
        # 原始路径
        w_out = self.W(x)
        
        # 空间路径
        s_down = self.spatial_down(x)
        s_out = self.spatial_up(s_down)
        
        # 频谱路径
        f_down = self.spectral_down(x) # (B, rank, H, W)
        f_gated = self.spectral_gate(f_down) # FFT -> Gate -> IFFT
        f_out = self.spectral_up(f_gated)
        
        # 融合输出
        adapter_out = (self.alpha_spatial * s_out) + (self.alpha_spectral * f_out)
        
        return w_out + adapter_out * self.lora_scale


# =============================================================================
# zw [创新点 3 集成] 时间感知双域适配器 (TimeAwareDualAdapter)
# 集成了以下创新点：
#   - 创新点 1: 频域门控 (SpectralGating)
#   - 创新点 2: 跨域交互 (CrossDomainInteraction)
#   - 创新点 3: 时间感知动态门控 (Time-Aware Gating)
# 根据扩散时间步 t 动态调整频域和空域的融合权重
# =============================================================================
class TimeAwareDualAdapter(nn.Module):
    """
    时间感知双域适配器
    集成了 Dual-Domain, CrossDomainInteraction 和 Time-Gating
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        rank: LoRA 秩
        lora_scale: LoRA 缩放因子
        t_emb_dim: 时间嵌入维度 (SD2.1 通常为 1280)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        rank: int = 16,
        lora_scale: float = 1.0,
        t_emb_dim: int = 1280  # SD2.1 的时间嵌入维度
    ):
        super().__init__()
        self.lora_scale = lora_scale
        self.t_emb_dim = t_emb_dim

        # --- 原始路径 (冻结) ---
        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters():
            p.requires_grad_(False)

        # --- 降维路径 ---
        # 1. 空间路径 (Spatial): 使用标准卷积捕捉局部特征
        self.spatial_down = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        # 2. 频谱路径 (Spectral): 使用 1x1 卷积配合 stride 降维，随后进 FFT
        self.spectral_down = nn.Conv2d(in_channels, rank, 1, stride, 0, bias=False)

        # --- 核心处理模块 ---
        # 创新点 1: 频域门控
        self.spectral_gate = SpectralGating(rank)
        # 创新点 2: 跨域交互
        self.interaction = CrossDomainInteraction(rank)

        # --- 升维路径 ---
        self.spatial_up = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)
        self.spectral_up = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)

        # --- 时间感知门控 (Time-Aware Gating) ---
        # 根据 t_emb 动态预测双域融合权重
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, 2)
        )

        # --- 初始化 ---
        nn.init.kaiming_uniform_(self.spatial_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.spatial_up.weight)
        nn.init.kaiming_uniform_(self.spectral_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.spectral_up.weight)

    def forward(self, x, t_emb=None):
        """
        前向传播
        
        Args:
            x: 输入特征图 (B, C, H, W)
            t_emb: 时间步嵌入向量 (B, t_emb_dim)。如果为 None，则尝试从全局变量获取。
        
        Returns:
            融合后的输出特征
        """
        # 尝试从全局变量获取时间嵌入
        if t_emb is None:
            t_emb = CURRENT_TIME_EMB

        # 1. 原始权重输出
        w_out = self.W(x)

        # 2. 降维
        s_feat = self.spatial_down(x)
        f_feat = self.spectral_down(x)

        # 3. 频域处理 (创新点 1)
        f_feat = self.spectral_gate(f_feat)

        # 4. 跨域交互 (创新点 2)
        s_refined, f_refined = self.interaction(s_feat, f_feat)

        # 5. 升维
        s_out = self.spatial_up(s_refined)
        f_out = self.spectral_up(f_refined)

        # 6. 动态融合 (创新点 3: 时间感知门控)
        if t_emb is not None:
            # 预测权重: (B, 2) -> 分别获取空域和频域的权重
            weights = self.time_mlp(t_emb)  # (B, 2)
            # 使用 Sigmoid * 2 将权重限制在 [0, 2] 之间
            alpha_s = (torch.sigmoid(weights[:, 0]) * 2.0).view(-1, 1, 1, 1)
            alpha_f = (torch.sigmoid(weights[:, 1]) * 2.0).view(-1, 1, 1, 1)
        else:
            # 兼容性 Fallback: 使用静态权重
            alpha_s, alpha_f = 1.0, 1.0

        adapter_out = (alpha_s * s_out) + (alpha_f * f_out)

        return w_out + adapter_out * self.lora_scale

## diff2Flow 深度估计增强版实施指南
项目代号: PG-FAFM (Physics-Guided & Frequency-Adaptive Flow Matching)

本文档详细描述了如何在 Diff2Flow 框架中实现以下四个核心创新点：

Dual-Domain Adapter: 双域（频域+空域）流形适配器。

Cross-Domain Interaction: 跨域交互机制。

Time-Aware Gating: 时间感知动态门控。

Geometric Consistency Loss: 几何一致性损失（融合频域感知）。

1. 核心模块实现 (lora.py)
本节代码应添加到 diff2flow/lora.py 或同等的适配器定义文件中。

1.1 前置组件 (SpectralGating & CrossDomainInteraction)
这些组件是主适配器的基础构建块。

Python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralGating(nn.Module):
    """
    [创新点 1 核心组件] 频域门控模块
    利用 FFT 实现全局特征混合，捕捉长距离依赖。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 初始化复数权重，scale 很小以保证初始训练稳定性
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 2D 实数 FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 频域滤波 (复数乘法)
        weight = torch.view_as_complex(self.complex_weight).view(1, C, 1, 1)
        x_fft = x_fft * weight
        
        # 3. 2D IFFT 还原
        x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        return x

class CrossDomainInteraction(nn.Module):
    """
    [创新点 2 核心组件] 跨域交互模块
    解决空域和频域特征割裂问题，通过 Attention Map 进行相互校准。
    """
    def __init__(self, dim):
        super().__init__()
        # 降维生成门控系数，降低计算量
        self.reduce = nn.Conv2d(dim * 2, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial_f, spectral_f):
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
1.2 主适配器类 (TimeAwareDualAdapter)
这是替换标准 LoRA 的核心类。

重要注意事项：

t_emb 的传入：标准的 nn.Conv2d forward 只接受 x。为了实现 Time-Aware，我们需要在调用此层时传入 t_emb（时间嵌入向量）。这需要修改 UNet 的调用逻辑（详见第 3 节）。

Python

class TimeAwareDualAdapter(nn.Module):
    """
    [创新点 3 集成] 时间感知双域适配器
    集成了 Dual-Domain, Interaction 和 Time-Gating。
    """
    def __init__(
        self,
        in_channels, out_channels, kernel_size, stride, padding,
        rank=16, lora_scale=1.0,
        t_emb_dim=1280 # ⚠️ 注意: 需根据 SD 模型配置调整 (SD2.1通常为1280)
    ):
        super().__init__()
        self.lora_scale = lora_scale

        # --- 原始路径 (冻结) ---
        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters(): p.requires_grad_(False)

        # --- 降维路径 ---
        # 1. 空间路径 (Spatial)
        self.spatial_down = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        # 2. 频谱路径 (Spectral): 使用 1x1 卷积配合 stride 降维，随后进 FFT
        self.spectral_down = nn.Conv2d(in_channels, rank, 1, stride, 0, bias=False)

        # --- 核心处理 ---
        self.spectral_gate = SpectralGating(rank)
        self.interaction = CrossDomainInteraction(rank)

        # --- 升维路径 ---
        self.spatial_up = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)
        self.spectral_up = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)

        # --- 时间感知门控 (Time-Aware Gating) ---
        # 根据 t_emb 动态预测融合权重
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
        Args:
            x: 输入特征图
            t_emb: 时间步嵌入向量 (Batch, t_emb_dim)。如果为 None，则退化为静态权重。
        """
        # 1. 原始权重输出
        w_out = self.W(x)

        # 2. 降维
        s_feat = self.spatial_down(x)
        f_feat = self.spectral_down(x)

        # 3. 频域处理
        f_feat = self.spectral_gate(f_feat)
        
        # 4. 跨域交互 (创新点 2)
        s_refined, f_refined = self.interaction(s_feat, f_feat)

        # 5. 升维
        s_out = self.spatial_up(s_refined)
        f_out = self.spectral_up(f_refined)

        # 6. 动态融合 (创新点 3)
        if t_emb is not None:
            # 预测权重: (B, 2) -> (B, 2, 1, 1)
            weights = self.time_mlp(t_emb).view(-1, 2, 1, 1)
            # 使用 Sigmoid * 2 将权重限制在 [0, 2] 之间，初始值容易平衡
            alpha_s = torch.sigmoid(weights[:, 0]) * 2.0
            alpha_f = torch.sigmoid(weights[:, 1]) * 2.0
        else:
            # 兼容性 Fallback
            alpha_s, alpha_f = 1.0, 1.0

        adapter_out = (alpha_s * s_out) + (alpha_f * f_out)

        return w_out + adapter_out * self.lora_scale
2. 损失函数实现 (training_losses)
本节代码应修改 Flow Matching 模型的主类文件（通常是 models/flow_matching.py 或类似文件）。

实现步骤：

在模型的 __init__ 中注册 Sobel 算子（避免每次 forward 都创建 Tensor，节省开销）。

添加 get_surface_normal 辅助方法。

重写 training_losses。

2.1 初始化注册 (在 __init__ 中)
Python

# 在 Model 类的 __init__ 方法中添加：
self.use_freq_loss = True # 开关

# 预定义 Sobel 算子用于几何损失，使用 register_buffer 保证设备同步且不参与梯度更新
sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
self.register_buffer('sobel_x', sobel_x)
self.register_buffer('sobel_y', sobel_y)
2.2 辅助函数与核心 Loss 逻辑
Python

    def get_surface_normal(self, img):
        """
        [辅助函数] 从流场/图像计算表面法向量
        Args:
            img: (B, C, H, W)
        Returns:
            normal: (B, 3, H, W) 归一化法向量
        """
        # 确保输入是单通道用于梯度计算 (如果是多通道流场，取平均或主通道)
        if img.shape[1] > 1:
            img_gray = img.mean(dim=1, keepdim=True)
        else:
            img_gray = img

        # 使用预注册的 buffer 计算梯度
        grad_x = F.conv2d(img_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(img_gray, self.sobel_y, padding=1)
        
        # 构造法向量 (-dz/dx, -dz/dy, 1)
        z_component = torch.ones_like(grad_x)
        normal = torch.cat([-grad_x, -grad_y, z_component], dim=1)
        
        # 归一化
        return F.normalize(normal, dim=1)

    def training_losses(self, x1: torch.Tensor, x0: torch.Tensor = None, **cond_kwargs):
        """
        [修改版] 包含频域感知与几何一致性损失
        """
        if x0 is None:
            x0 = torch.randn_like(x1)

        bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

        # 采样时间 t
        t = torch.rand(bs, device=dev, dtype=dtype)

        # 计算流场样本
        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)
        vt = self.sample_vt(fm_x=xt, fm_t=t, **cond_kwargs)

        # --- Part A: 频域感知损失 (Frequency-Aware Loss) ---
        # 分离低频 (Low) 和 高频 (High)
        vt_low = F.interpolate(
            F.avg_pool2d(vt, kernel_size=4, stride=4), 
            size=vt.shape[-2:], mode='bilinear', align_corners=False
        )
        ut_low = F.interpolate(
            F.avg_pool2d(ut, kernel_size=4, stride=4), 
            size=ut.shape[-2:], mode='bilinear', align_corners=False
        )
        
        vt_high = vt - vt_low
        ut_high = ut - ut_low

        # 动态权重 w(t): 
        # t->0 (去噪后期): 此时应关注低频结构，w_low 变大
        # t->1 (去噪初期): 此时应关注高频细节，w_high 变大
        # 注意: 这里的 t 定义取决于 Diff2Flow 具体实现 (t=0是数据还是噪声)。
        # 假设 t=0 是数据，t=1 是噪声：
        # 则 t 接近 0 时 (生成最后阶段)，我们需要精细的几何结构，理应重点关注 Geometry Loss。
        t_w = t.view(-1, 1, 1, 1)
        w_low = 2.0 - t_w
        w_high = 1.0 + t_w
        
        loss_pixel = (w_low * (vt_low - ut_low).square() + w_high * (vt_high - ut_high).square()).mean()

        # --- Part B: 几何一致性损失 (Geometry Consistency Loss) [创新点 4] ---
        # 强制预测流场 vt 的梯度结构与目标流场 ut 一致
        pred_norm = self.get_surface_normal(vt)
        target_norm = self.get_surface_normal(ut)
        
        # Cosine Loss: 1 - mean(cos_sim)
        loss_geo = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
        
        # 几何损失权重，建议 0.1 ~ 0.2
        lambda_geo = 0.2
        
        return loss_pixel + lambda_geo * loss_geo
3. 集成与调用指南 (Integration Strategy)
这是最容易出错的部分，特别是 t_emb 的传递。

3.1 注入 Time Embedding (model.py / unet.py)
Diff2Flow 通常使用替换 nn.Conv2d 的方式插入 LoRA。你需要找到替换逻辑或 UNet 的 Forward 循环。

方案 A：修改 UNet Forward (推荐) 你需要修改 UNet 的代码，使其在遍历层时，如果是 TimeAwareDualAdapter 类型，显式传入 t_emb。

Python

# 伪代码示例：在 UNet 的 forward 函数中
# t_emb 通常由 timestep t 经过 TimeEmbedding 层得到
t_emb = self.time_embed(t) 

for name, module in self.named_modules():
    if isinstance(module, TimeAwareDualAdapter):
        # 这是一个 Hack，因为通常我们调用 model(x)。
        # 我们需要在 model 定义外部或者使用 hook 机制。
        pass
方案 B：使用 Context Manager (工程上最简便) 如果不想修改复杂的 UNet 递归调用结构，可以使用全局上下文变量。

在 lora.py 中添加全局变量：

Python

CURRENT_TIME_EMB = None
在 Model 的 training_losses 或 forward 开始处设置：

Python

import lora
# ... 获取 t_emb 后
lora.CURRENT_TIME_EMB = t_emb
# 执行网络
vt = self.sample_vt(...)
# 清理
lora.CURRENT_TIME_EMB = None
在 TimeAwareDualAdapter.forward 中读取：

Python

def forward(self, x, t_emb=None):
    if t_emb is None:
        # 尝试从全局变量获取
        import lora
        t_emb = lora.CURRENT_TIME_EMB
    # ... 继续后续逻辑
3.2 配置文件修改 (configs/*.yaml)
修改 Hydra 配置文件以启用新的 Adapter。

YAML

# configs/experiment/img2depth/obj_base.yaml

lora:
  lora_cfg:
    # 确保 lora.py 中有映射将此字符串指向 TimeAwareDualAdapter 类
    lora_type: time_aware_dual 
    rank: 16
    # 传递给 Adapter 的参数
    t_emb_dim: 1280 
4. 实验验证清单
在实施后，请按以下顺序验证：

尺寸检查 (Shape Check): 打印 x_fft 和 gate 的 shape，确保频域变换和通道交互没有维度错误。

梯度检查 (Grad Check): 运行 1 个 step，检查 loss_geo 是否有梯度回传 (loss_geo.backward())。

过拟合测试: 使用一张图片训练，观察深度图是否能迅速收敛且边缘锐利（几何损失生效的标志）。
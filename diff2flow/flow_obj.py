import math
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
import numpy as np

from diff2flow.flow import FlowModel
from diff2flow.flow import forward_with_cfg

from diff2flow.utils.diffusion_utils import make_beta_schedule
from diff2flow.utils.diffusion_utils import enforce_zero_terminal_snr
from diff2flow.utils.diffusion_utils import extract_and_interpolate_into_tensor as extract_into_tensor

import torch.nn.functional as F

""" Flow Model """


class FlowModelObj(FlowModel):
    def __init__(
        self,
        enforce_zero_snr: bool = True,
        diffusion_parameterization: str = 'v',
        diffusion_schedule: str = 'linear',
        use_freq_loss: bool = True, # 添加控制开关
        # zw [创新点 4] 几何一致性损失开关和权重
        use_geo_loss: bool = True,
        lambda_geo: float = 0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.register_sdv2_schedule(diffusion_schedule, enforce_zero_snr)
        assert diffusion_parameterization in ['v', 'eps'], 'Diffusion parameterization has to be either v or eps'
        self.diffusion_parameterization = diffusion_parameterization
        self.diffusion_schedule = diffusion_schedule
        self.use_freq_loss = use_freq_loss
        # zw [创新点 4] 初始化几何一致性损失参数
        self.use_geo_loss = use_geo_loss
        self.lambda_geo = lambda_geo
        
        # zw [创新点 4] 注册 Sobel 算子用于几何损失
        # 使用 register_buffer 保证设备同步且不参与梯度更新
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def ode_fn(self, t, x, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        _pred = self.sample_vt(x, t, **kwargs)
        return _pred
    
    def register_sdv2_schedule(self, diffusion_schedule, enforce_zero_snr=True):
        # SDV2 schedule
        linear_start = 0.00085
        linear_end = 0.0120

        betas = make_beta_schedule(
            diffusion_schedule,
            n_timestep=1000,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        if enforce_zero_snr:
            betas = enforce_zero_terminal_snr(betas)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_full = np.append(1., alphas_cumprod)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        # self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('alphas_cumprod_full', to_torch(alphas_cumprod_full))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_alphas_cumprod_full', to_torch(np.sqrt(alphas_cumprod_full)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_full', to_torch(np.sqrt(1. - alphas_cumprod_full)))

        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        self.register_buffer('rectified_alphas_cumprod_full', self.sqrt_alphas_cumprod_full / (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full))
        self.register_buffer('rectified_sqrt_alphas_cumprod_full', self.sqrt_one_minus_alphas_cumprod_full / (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full))

    def sample_vt(self, fm_x, fm_t, **kwargs):
        """
        Sample the v-parameterized vector field at time t
        """
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        # print(fm_t, dm_t)
        dm_x = self.convert_fm_xt_to_dm_xt(fm_x, fm_t)
        # vt = self.net(dm_x, dm_t, **kwargs)
        vt = forward_with_cfg(dm_x, dm_t, self.net, **kwargs)
        
        # TODO: ugly fix for nan values!!!
        if torch.isnan(vt).any():
            vt[torch.isnan(vt)] = 0
        
        # vt = self.forward(x=dm_x, t=dm_t, **kwargs)
        if self.diffusion_parameterization == 'v':
            vector_field = self.get_vector_field_from_v(vt, dm_x, dm_t)
        elif self.diffusion_parameterization == 'eps':
            vector_field = self.get_vector_field_from_eps(vt, dm_x, dm_t)
        return vector_field

    def convert_fm_t_to_dm_t(self, t):
        """
        Convert the continuous time t in [0,1] to discrete time t [0, 1000]
        # TODO: Make it compatible with zero-terminal SNR
        """
        rectified_alphas_cumprod_full = self.rectified_alphas_cumprod_full.clone().to(t.device)
        # reverse the rectified_alphas_cumprod_full for searchsorted
        rectified_alphas_cumprod_full = torch.flip(rectified_alphas_cumprod_full, [0])
        right_index = torch.searchsorted(rectified_alphas_cumprod_full, t, right=True)
        left_index = right_index - 1
        right_value = rectified_alphas_cumprod_full[right_index]
        left_value = rectified_alphas_cumprod_full[left_index]
        dm_t = left_index + (t - left_value) / (right_value - left_value)
        # now reverse back the dm_t
        dm_t = self.num_timesteps - dm_t
        return dm_t
    
    def convert_fm_xt_to_dm_xt(self, fm_xt, fm_t):
        """
        Convert fm trajectory to dm trajectory using the fm t
        We use linear scaling here
        """
        scale = self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        # do lienar interpolation here
        dm_t_left_index = torch.floor(dm_t)
        dm_t_right_index = torch.ceil(dm_t)
        dm_t_left_value = scale[dm_t_left_index.long()]
        dm_t_right_value = scale[dm_t_right_index.long()]

        scale_t = dm_t_left_value + (dm_t - dm_t_left_index) * (dm_t_right_value - dm_t_left_value)
        scale_t = scale_t.view(-1, 1, 1, 1)
        dm_xt = fm_xt * scale_t
        return dm_xt

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )
    
    def predict_start_from_eps(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def get_vector_field_from_v(self, v, x_t, t):
        """
        v is the SD v-parameterized vector field with v = sqrt(alpha_cumprod) * eps - sqrt(1 - alpha_cumprod) * z
        the FM vector field is defined as z - eps

        First of all convert the x_t from the rectified flow trajectory to the original diffusion trajectory
        Then calculate the vector field from the v-parameterized vector field
        """
        z_pred = self.predict_start_from_z_and_v(x_t, t, v)
        eps_pred = self.predict_eps_from_z_and_v(x_t, t, v)
        vector_field = z_pred - eps_pred                    # z - eps
        return vector_field
    
    def get_vector_field_from_eps(self, noise, x_t, t):
        """
        eps is the SD eps-parameterized vector field with
        the FM vector field is defined as z - eps

        First of all convert the x_t from the rectified flow trajectory to the original diffusion trajectory
        Then calculate the vector field from the eps-parameterized vector field
        """
        z_pred = self.predict_start_from_eps(x_t, t, noise)
        eps_pred = noise
        vector_field = z_pred - eps_pred                    # z - eps
        return vector_field
    
    def forward(self, x, t, **kwargs):
        """
        Forward pass for the flow model
        """
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        _pred = self.sample_vt(x, t, **kwargs)
        return _pred


    # def training_losses(self, x1: Tensor, x0: Tensor = None, **cond_kwargs):
    #     """
    #     Args:
    #         x1: shape (bs, *dim), represents the target minibatch (data)
    #         x0: shape (bs, *dim), represents the source minibatch, if None
    #             we sample x0 from a standard normal distribution.
    #         cond_kwargs: additional arguments for the conditional flow
    #             network (e.g. conditioning information)
    #     Returns:
    #         loss: scalar, the training loss for the flow model
    #     """
    #     if x0 is None:
    #         x0 = torch.randn_like(x1)

    #     bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

    #     # Sample time t from uniform distribution U(0, 1)
    #     t = torch.rand(bs, device=dev, dtype=dtype)

    #     # sample xt and ut
    #     xt = self.compute_xt(x0=x0, x1=x1, t=t)
    #     ut = self.compute_ut(x0=x0, x1=x1, t=t)
    #     vt = self.sample_vt(fm_x=xt, fm_t=t, **cond_kwargs)

    #     return (vt - ut).square()

    # =============================================================================
    # zw [创新点 4 辅助函数] 计算表面法向量
    # 从流场/图像计算表面法向量，用于几何一致性损失
    # =============================================================================
    def get_surface_normal(self, img):
        """
        从流场/图像计算表面法向量
        
        Args:
            img: (B, C, H, W) 输入图像或流场
            
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

    def training_losses(self, x1: Tensor, x0: Tensor = None, **cond_kwargs):
        """
        Args:
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch, if None
                we sample x0 from a standard normal distribution.
            cond_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            loss: scalar, the training loss for the flow model
        """
        if x0 is None:
            x0 = torch.randn_like(x1)

        bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

        # Sample time t from uniform distribution U(0, 1)
        t = torch.rand(bs, device=dev, dtype=dtype)

        # sample xt and ut
        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)
        vt = self.sample_vt(fm_x=xt, fm_t=t, **cond_kwargs)

        if not self.use_freq_loss:
            return (vt - ut).square()

        # =============================================================================
        # zw [创新点: 频域感知损失] Frequency-Aware Loss
        # 使用平均池化提取低频分量 (Low-frequency)
        # 4x4 窗口可以有效平滑图像，提取宏观几何结构
        # =============================================================================
        vt_low = F.interpolate(
            F.avg_pool2d(vt, kernel_size=4, stride=4), 
            size=vt.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        ut_low = F.interpolate(
            F.avg_pool2d(ut, kernel_size=4, stride=4), 
            size=ut.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 提取高频分量 (High-frequency: Original - Low)
        vt_high = vt - vt_low
        ut_high = ut - ut_low

        # 动态时间权重: 
        # t -> 0 (靠近噪声): 重点优化低频结构 (w_low 较高)
        # t -> 1 (靠近数据): 重点优化高频细节 (w_high 较高)
        t_w = t.view(-1, 1, 1, 1)
        w_low = 2.0 - t_w   # 范围 [2.0, 1.0]
        w_high = 1.0 + t_w  # 范围 [1.0, 2.0]

        # 计算加权平方损失 (频域感知像素损失)
        loss_pixel = w_low * (vt_low - ut_low).square() + w_high * (vt_high - ut_high).square()

        # =============================================================================
        # zw [创新点 4] 几何一致性损失 (Geometry Consistency Loss)
        # 强制预测流场 vt 的梯度结构与目标流场 ut 一致
        # 通过比较法向量的余弦相似度来实现
        # =============================================================================
        if self.use_geo_loss:
            pred_norm = self.get_surface_normal(vt)
            target_norm = self.get_surface_normal(ut)
            
            # Cosine Loss: 1 - mean(cos_sim)
            # 法向量越一致，余弦相似度越接近 1，损失越小
            loss_geo = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
            
            # 返回总损失: 像素损失 + 几何一致性损失
            return loss_pixel + self.lambda_geo * loss_geo
        else:
            return loss_pixel
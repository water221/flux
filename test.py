import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from omegaconf import OmegaConf
import hydra
from pytorch_lightning import Trainer, seed_everything
from diff2flow.helpers import instantiate_from_config, exists
from diff2flow.trainer_module import TrainerModuleLatentFM

import einops
import numpy as np
from PIL import Image
import wandb
from diff2flow.visualizer import per_sample_min_max_normalization

class PatchedTrainer(TrainerModuleLatentFM):
    def validation_step(self, batch, batch_idx):
        # Inject dummy conditioning for CrossAttention (1024 dim)
        # We use a key "dummy_cond" and ensure it's available
        # x1 is the target image, so we use its batch size
        batch_size = batch["x1"].shape[0]
        batch["dummy_cond"] = torch.zeros(batch_size, 1, 1024, device=self.device)
        return super().validation_step(batch, batch_idx)

    def evaluate_and_visualize_batch(self, batch, prefix="train"):
        # Override to save images to disk instead of using logger
        data = self.extract_from_batch(batch)
        x1, x1_latent = data["x1"], data["x1_latent"]
        x0, x0_latent = data["x0"], data["x0_latent"]
        
        # concatenated context
        if exists(self.context_key):
            context = data[self.context_key]
        else:
            context = None

        # input
        if self.start_from_noise:
            x_source = batch.get("noise", torch.randn_like(x1_latent))
        else:
            x_source = x0_latent

        # noise x0
        if self.noise_image:
            x_source = self.diffusion.q_sample(x_start=x_source, t=self.noising_step)

        # prediction
        model = self.ema_model if self.use_ema_for_sampling else self.model
        sample_kwargs = dict(num_steps=self.sampling_steps) if hasattr(self, "sampling_steps") else {}
        
        # Inject dummy cond for inference too
        batch_size = x_source.shape[0]
        dummy_cond = torch.zeros(batch_size, 1, 1024, device=self.device)
        
        x1_latent_pred = model.generate(x_source, context=context, context_ca=dummy_cond, sample_kwargs=sample_kwargs)
        x1_pred = self.decode_first_stage(x1_latent_pred)

        # visualize
        if self.visualizer is not None:
            print("Generating visualization...")
            
            # Get filenames if available
            filenames = batch.get("__key__", [f"{prefix}_{self.global_step}_{i}" for i in range(len(x0))])
            
            # Iterate over batch to save separately
            for i in range(len(x0)):
                filename = os.path.basename(filenames[i])
                
                # 1. Save Comparison (Input, GT, Pred)
                # Slice tensors to keep dimensions (1, C, H, W)
                x0_i = x0[i:i+1]
                x1_i = x1[i:i+1]
                x1_pred_i = x1_pred[i:i+1]
                
                comp_img = self.visualizer(x0_i, x1_i, x1_pred_i)
                comp_save_path = os.path.join(self.trainer.default_root_dir, f"{filename}.png")
                comp_img.save(comp_save_path)
                
                # 2. Save Prediction Separately
                # Normalize depth prediction for visualization (same logic as visualizer)
                pred_norm = per_sample_min_max_normalization(x1_pred_i)
                if pred_norm.shape[1] == 1:
                    pred_norm = pred_norm.repeat(1, 3, 1, 1)
                
                pred_img_tensor = pred_norm * 255
                pred_img_tensor = pred_img_tensor.clip(0, 255).byte()
                pred_img_np = pred_img_tensor[0].permute(1, 2, 0).cpu().numpy()
                pred_img = Image.fromarray(pred_img_np)
                
                pred_save_path = os.path.join(self.trainer.default_root_dir, f"{filename}_pred.png")
                pred_img.save(pred_save_path)
                
            print(f"Saved {len(x0)} images to {self.trainer.default_root_dir}")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    # 1. 设置环境
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)
    
    # 2. 实例化数据
    print(f"Loading Data: {cfg.data.target}")
    data = instantiate_from_config(cfg.data)
    # data.setup() # Lightning Trainer 会自动调用 setup

    # 3. 实例化模型
    # print(f"Loading Model: {cfg.model.target}") # cfg.model 结构可能不同，先注释掉
    
    # 确保 lora 配置被正确传递
    lora_cfg = None
    if "lora" in cfg and cfg.lora is not None:
        lora_cfg = cfg.lora.get("lora_cfg", None)

    # 构造 first_stage 配置 (Autoencoder)
    # TrainerModuleLatentFM 需要一个包含 scale_factor 和 first_stage_cfg 的配置对象
    first_stage_wrapper = None
    if "autoencoder" in cfg:
        print("Configuring First Stage (Autoencoder)...")
        # cfg.autoencoder 已经包含了 scale_factor 和 first_stage_cfg
        # 所以我们不需要再包装一层，直接传给 TrainerModuleLatentFM 即可
        # 但 TrainerModuleLatentFM 期望 first_stage 参数是一个对象，它有 first_stage_cfg 属性
        # 而 cfg.autoencoder 本身就是这个对象
        first_stage_wrapper = cfg.autoencoder

    # FIX: 动态修改 fm_cfg 中的 net_cfg 参数，开启 concat_context
    # 这是为了解决 RuntimeError: Given groups=1, weight of size [320, 8, 3, 3], expected input[4, 4, 48, 64] to have 8 channels
    # 因为 obj_ch8 配置期望 8 通道输入 (4ch Noise + 4ch Condition)，必须让 UNet 执行拼接
    if "net_cfg" in cfg.model.fm_cfg.params:
         print("Forcing concat_context=True for 8-channel input...")
         # 使用 OmegaConf.update 强制添加新键，因为默认是 struct 模式，不允许添加新键
         OmegaConf.set_struct(cfg, False)
         cfg.model.fm_cfg.params.net_cfg.params.concat_context = True
         # FIX: 必须指定 context_key，否则 context 为 None，拼接不会发生
         # 对于 Image-to-Depth，条件是源图像 Latent (x0_latent)
         cfg.model.context_key = "x0_latent"
         OmegaConf.set_struct(cfg, True)

    # Define dummy cond stage config for CrossAttention
    dummy_cond_stage_cfg = OmegaConf.create({"target": "torch.nn.Identity"})

    model = PatchedTrainer(
        fm_cfg=cfg.model.fm_cfg, # FIX: 传入 fm_cfg 子项，因为它包含 target
        lora_cfg=lora_cfg,
        # 传入可视化配置，确保能生成图片
        # FIX: visualizer 通常在 task 配置中，而不是 model 配置中
        visualizer=cfg.task.get("visualizer", None) if "task" in cfg else cfg.model.get("visualizer", None),
        # 这里的参数主要用于训练，测试时不太重要，但为了初始化不报错，传入一些默认值
        lr_scheduler_cfg=None,
        cond_dropout=0.0,
        # FIX: 传入正确的 first_stage 配置，以便模型能加载 VAE 并处理 RGB 输入
        first_stage=first_stage_wrapper,
        # FIX: 显式传入 context_key，否则 TrainerModuleLatentFM 内部默认为 None
        context_key="x0_latent",
        # FIX: 传入 dummy cond stage 和 key，解决 CrossAttention 维度不匹配问题
        cond_stage_cfg=dummy_cond_stage_cfg,
        conditioning_key="dummy_cond",
        # FIX: 传入 metric_tracker_cfg 以便计算指标
        metric_tracker_cfg=cfg.task.get("metric_tracker_cfg", None) if "task" in cfg else None,
    )
    # model.scale_factor = 1.0 # 不再需要手动设置，first_stage_wrapper 会处理

    # 4. 加载权重
    ckpt_path = cfg.resume_checkpoint
    if not exists(ckpt_path):
        print(f"Error: Please provide a valid resume_checkpoint path! Current: {ckpt_path}")
        # 尝试从 load_weights 加载，如果 resume_checkpoint 为空
        if exists(cfg.load_weights):
             ckpt_path = cfg.load_weights
             print(f"Falling back to load_weights: {ckpt_path}")
        else:
             return

    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint # 可能是纯权重文件
        
    # 加载权重，允许不匹配 (strict=False)
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")

    # 5. 配置 Trainer
    gpu_kwargs = {'accelerator': 'gpu', 'devices': 1}
    
    # 确保输出目录存在
    log_dir = cfg.get("log_dir", "logs/test_output")
    os.makedirs(log_dir, exist_ok=True)

    trainer = Trainer(
        **gpu_kwargs,
        logger=False, 
        enable_checkpointing=False,
        default_root_dir=log_dir,
        limit_val_batches=2, # FIX: 只跑 2 个 batch 以便快速看到结果
    )

    # 6. 运行验证
    print("Starting Validation...")
    trainer.validate(model, datamodule=data)

if __name__ == "__main__":
    main()

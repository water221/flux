模型lora训练
python train.py experiment=img2depth/obj_base data=hypersim lora=c0.2_fullA0.2_FL lora.lora_cfg.lora_type=dual_domain lora.lora_cfg.lora_conv=16


1. 根据训练后的模型进行测试
```
python train.py \
    experiment=img2depth/obj_base \
    data=hypersim \
    lora=c0.2_fullA0.2_FL \
    lora.lora_cfg.lora_type=dual_domain \
    lora.lora_cfg.lora_conv=16 \
    resume_checkpoint="logs/img2depth/obj/base/.../checkpoints/best.ckpt" \
    train.trainer_params.max_epochs=0 \
    train.trainer_params.num_sanity_val_steps=-1
```
resume_checkpoint: 指定你训练好的权重。
max_epochs=0: 不进行训练。
num_sanity_val_steps=-1: 运行完整的验证集循环。这样，Lightning 会加载模型，跳过训练，直接运行验证，并输出验证集指标和可视化结果。这是最快验证模型性能的方法。




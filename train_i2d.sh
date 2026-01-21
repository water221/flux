python train.py \
    experiment=img2depth/obj_base \
    data=hypersim \
    lora=c0.2_fullA0.2_FL \
    lora.lora_cfg.lora_type=dual_domain \
    lora.lora_cfg.lora_conv=16 \
    +model.use_freq_loss=True
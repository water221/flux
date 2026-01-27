python test.py \
    experiment=img2depth/obj_base \
    data=hypersim \
    lora=c0.2_fullA0.2_FL \
    lora.lora_cfg.lora_type=time_aware_dual \
    lora.lora_cfg.lora_conv=16 \
    resume_checkpoint="/home/zhutianshui/workspace/demoProjects/flux/logs/img2depth/obj/base/v0_2026-01-16-15-08-53/checkpoints/last.ckpt"
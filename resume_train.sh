python train.py \
    experiment=img2depth/obj_base \
    data=hypersim \
    lora=c0.2_fullA0.2_FL \
    lora.lora_cfg.lora_type=dual_domain \
    lora.lora_cfg.lora_conv=16 \
    resume_checkpoint="/home/zhutianshui/workspace/demoProjects/flux/logs/img2depth/obj/base/v0_2025-12-26-10-06-39/checkpoints/step100000.ckpt"
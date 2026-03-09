#!/bin/bash
# Exp H: 全層訓練 + 深層 L2 正則化
# 全部 18 層 LoRA，但對深層施加 L2 正則化限制變化

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python exp_0106/train.py \
    --exp_name exp_h_l2_reg \
    --model_type l2_reg \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lr 1e-4 \
    --l2_reg_weight 0.1 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_increment 0.1 \
    --phase_advance_epochs 30 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --num_epochs 300 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 2 \
    2>&1 | tee exp_0106/exp_h.log

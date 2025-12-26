#!/bin/bash
# Exp64: Curriculum Learning 實驗
#
# 核心概念：
# - 從簡單樣本 (高 SNR) 開始訓練，逐步增加難度 (低 SNR)
# - 讓模型先學會簡單的 denoising，再處理困難樣本
#
# Curriculum 配置：
# - Mode: curriculum (從易到難)
# - 初始使用 30% 最簡單的資料
# - 每 30 epoch 增加 10% 資料
# - 300 epoch 後會使用全部資料
#
# 其他配置與 Exp55 相同

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1226/train_exp64_curriculum.py \
    --exp_name exp64_curriculum \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1226/runs/exp64_curriculum \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_increment 0.1 \
    --phase_advance_epochs 30 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 300 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1226/exp64.log

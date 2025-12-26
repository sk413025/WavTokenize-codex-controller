#!/bin/bash
# Exp63: VQ-Aware Loss 實驗
#
# 核心概念：
# - 加入 VQ Commitment Loss: 讓 student features 更接近 VQ codebook centroids
# - 加入 VQ Distortion Loss: 最小化 soft-VQ 後的誤差
# - 減少 VQ quantization error，改善 decode 後的音質
#
# 配置：
# - 基礎: Feature + Triplet Loss (與 Exp55 相同)
# - 新增: VQ Commitment Loss (λ=0.1)
# - 新增: VQ Distortion Loss (λ=0.1)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1226/train_exp63_vq_aware.py \
    --exp_name exp63_vq_aware \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1226/runs/exp63_vq_aware \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --vq_commitment_weight 0.1 \
    --vq_distortion_weight 0.1 \
    --vq_temperature 1.0 \
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
    2>&1 | tee exp_1226/exp63.log

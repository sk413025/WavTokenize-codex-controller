#!/bin/bash
# Exp67: Curriculum Learning + VQ-Aware Loss 組合
#
# 結合 Exp64 (最佳) 和 Exp63 (有改善) 的優點：
# - Curriculum Learning: 從簡單樣本開始
# - VQ-Aware Loss: 改善 VQ 量化品質
#
# 基於 Exp64 (Best Val Acc: 1.06%) 配置

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1226/train_exp67_curriculum_vq.py \
    --exp_name exp67_curriculum_vq \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1226/runs/exp67_curriculum_vq \
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
    --early_stopping_patience 100 \
    2>&1 | tee exp_1226/exp67.log

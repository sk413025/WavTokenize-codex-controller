#!/bin/bash
# Exp69: Anti-Collapse 輕量版
#
# Exp65 問題分析：
# - 正則化太強 (entropy=0.1, diversity=0.1, contrastive=0.1)
# - 模型學習能力被抑制
# - Best Val Acc 只有 0.69% (比 baseline 0.91% 還差)
#
# 修復方案：
# - 大幅降低正則化強度
# - entropy_weight: 0.1 -> 0.01
# - diversity_weight: 0.1 -> 0.01
# - contrastive_weight: 0.1 -> 0.01
# - 保持 Frame-Tolerant 功能

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python families/compat_legacy/curriculum_data/train_exp65_anti_collapse.py \
    --exp_name exp69_anti_collapse_light \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/families/compat_legacy/curriculum_data/runs/exp69_anti_collapse_light \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --entropy_weight 0.01 \
    --diversity_weight 0.01 \
    --contrastive_weight 0.01 \
    --diversity_margin 0.5 \
    --contrastive_temperature 0.1 \
    --use_frame_tolerant \
    --frame_tolerance 1 \
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
    2>&1 | tee families/compat_legacy/curriculum_data/exp69.log

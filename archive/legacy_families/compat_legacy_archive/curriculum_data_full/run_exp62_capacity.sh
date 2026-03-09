#!/bin/bash
# Exp62: 擴大容量實驗
#
# 基於 Exp55 (最佳 baseline, 0.91% Val Acc)
# 擴大模型容量：
# - LoRA rank: 256 -> 384
# - Epochs: 200 -> 500
# - Early stopping: patience=50
#
# 目標：測試更大容量是否能突破 Val Acc 瓶頸

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python families/compat_legacy/curriculum_data/train_exp62_capacity.py \
    --exp_name exp62_capacity \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/families/compat_legacy/curriculum_data/runs/exp62_capacity \
    --lora_rank 384 \
    --lora_alpha 768 \
    --lora_dropout 0.25 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 500 \
    --early_stopping_patience 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    2>&1 | tee families/compat_legacy/curriculum_data/exp62.log

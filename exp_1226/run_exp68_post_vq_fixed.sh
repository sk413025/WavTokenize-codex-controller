#!/bin/bash
# Exp68: Post-VQ Feature Loss 修復版
#
# Exp66 崩潰原因分析：
# - Post-VQ loss weight 太大 (0.5)
# - Straight-Through Estimator 梯度不穩定
# - Epoch 20 後 Post-VQ Cos Sim 變成負數
#
# 修復方案：
# - 降低 post_vq_weight: 0.5 -> 0.05
# - 降低 post_vq_cosine_weight: 0.5 -> 0.05
# - 保持基礎 Feature + Triplet Loss 為主

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1226/train_exp66_post_vq.py \
    --exp_name exp68_post_vq_fixed \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1226/runs/exp68_post_vq_fixed \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --post_vq_feature_weight 0.05 \
    --post_vq_cosine_weight 0.05 \
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
    2>&1 | tee exp_1226/exp68.log

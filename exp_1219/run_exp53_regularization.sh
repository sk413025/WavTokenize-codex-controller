#!/bin/bash
# Exp53: 增強正則化實驗
#
# 基於 Exp48 的過擬合診斷:
# - Train Loss 持續下降 (1.91 → 1.07)
# - Val Loss 在 epoch 12 後停滯 (~1.12)
# - Train/Val Acc gap 持續擴大
#
# 修改:
# - lora_dropout: 0.2 -> 0.4 (加強 dropout)
# - weight_decay: 0.05 -> 0.1 (加強 L2 正則化)
#
# 預期:
# - 縮小 Train/Val gap
# - 可能犧牲一些 Train Acc，但提升 Val Acc

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1219/train.py \
    --exp_name exp53_regularization \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.4 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.1 \
    --batch_size 16 \
    --num_epochs 200 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1219/exp53.log

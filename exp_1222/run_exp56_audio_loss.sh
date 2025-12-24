#!/bin/bash
# Exp56: Audio Domain Loss 基線實驗
#
# 核心改變:
# - 使用 STFT + Mel Loss 直接優化音頻品質
# - Bypass VQ: Encoder features → Decode (不經過離散化)
# - 不使用 Feature Loss / Triplet Loss
#
# 預期:
# - 音頻品質 (PESQ/STOI) 大幅提升
# - Token Accuracy 可能不變（因為不直接優化）

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1222/train.py \
    --exp_name exp56_audio_loss \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --feature_weight 0.0 \
    --triplet_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 5 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1222/exp56.log

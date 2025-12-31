#!/bin/bash
# Exp76: Two-Stage Training
#
# 核心改進：
# Stage 1: 先訓練 waveform-level 去噪 (不經過 VQ)
# Stage 2: 用去噪後的音訊去匹配 VQ tokens

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Exp76: Two-Stage Training"
echo "=============================================="
echo "Stage 1: Waveform Denoising (100 epochs)"
echo "Stage 2: Token Matching (200 epochs)"
echo "=============================================="

python -u exp_1231/train_exp76_two_stage.py \
    --exp_name exp76_two_stage \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --stage1_epochs 100 \
    --stage1_lr 1e-4 \
    --stage2_epochs 200 \
    --stage2_lr 5e-5 \
    --freeze_stage1 \
    --batch_size 8 \
    --num_workers 4 \
    --use_amp \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 2 \
    --seed 42

echo "=============================================="
echo "Exp76 completed!"
echo "=============================================="

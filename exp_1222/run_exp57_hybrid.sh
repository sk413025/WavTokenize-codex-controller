#!/bin/bash
# Exp57: Hybrid Loss (Audio + Feature)
#
# 結合兩種 Loss:
# - Audio Loss: STFT + Mel (主要，優化音質)
# - Feature Loss: MSE + Triplet (輔助，幫助 VQ 選對)
#
# 目的:
# - 既有好音質，又提升 Token Accuracy
# - 讓模型學會產生「既能重建好音質，又接近正確 token」的 features

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1222/train.py \
    --exp_name exp57_hybrid \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --feature_weight 0.1 \
    --triplet_weight 0.1 \
    --triplet_margin 0.2 \
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
    2>&1 | tee exp_1222/exp57.log

#!/bin/bash
# Exp58: 微調 Exp48 模型 (從最佳 Token Accuracy 模型開始)
#
# 策略:
# - 先用 Token-level loss 訓練到最佳 Token Accuracy (Exp48)
# - 再用 Audio Loss 微調，提升音質
#
# 預期:
# - 保持較高的 Token Accuracy
# - 同時提升音頻品質

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 從 Exp48 最佳模型開始
python exp_1222/train.py \
    --exp_name exp58_finetune_exp48 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --feature_weight 0.05 \
    --triplet_weight 0.05 \
    --triplet_margin 0.2 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 3 \
    --grad_clip 1.0 \
    --resume ../exp_1217/runs/exp48_best_config/best_model.pt \
    2>&1 | tee exp_1222/exp58.log

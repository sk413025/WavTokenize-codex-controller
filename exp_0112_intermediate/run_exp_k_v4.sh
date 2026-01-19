#!/bin/bash
# Exp K v4: Optimized Intermediate Layer Supervision
#
# 改進重點:
# 1. 移除 L10 監督 (效果存疑)
# 2. L5 權重提高到 1.0 (收斂最好)
# 3. L6 權重降低到 0.5 (避免過擬合)
# 4. 總權重降低到 0.5
# 5. weight_decay 提高到 0.1

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 設定 GPU
export CUDA_VISIBLE_DEVICES=0

# 執行 v4 訓練
python exp_0112_intermediate/train_v4.py \
    --exp_name exp_k_v4 \
    --seed 42 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --intermediate_weight 0.5 \
    --intermediate_L3_weight 0.3 \
    --intermediate_L5_weight 1.0 \
    --intermediate_L6_weight 0.5 \
    --num_epochs 300 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 0.1 \
    --curriculum_start 0.3 \
    --curriculum_end 1.0 \
    --curriculum_epochs 100 \
    --save_audio_interval 50 \
    --use_amp \
    2>&1 | tee exp_0112_intermediate/exp_k_v4.log

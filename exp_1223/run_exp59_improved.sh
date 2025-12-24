#!/bin/bash
# Exp59: 基於 Exp55 最佳配置的改進實驗
#
# 改進內容 (相對於 Exp55):
# 1. 學習率: 1e-4 → 5e-5 (更穩定)
# 2. Gradient Accumulation: 2 → 4 (等效 batch size: 8×4=32)
# 3. 基於 Val Accuracy 保存最佳模型 (train.py 已實現)
# 4. 不使用 Early Stopping (訓練完整 300 epochs)
#
# Exp55 配置回顧:
# - rank=256, alpha=512, dropout=0.2
# - batch_size=8, grad_accum=2, 等效 batch=16
# - lr=1e-4
# - Best Val Acc: 0.91% at epoch 173
# - 問題: Val Loss 在 epoch 56 後過擬合 (+4.49%)
#
# Exp59 改進目標:
# - 使用更小的學習率減緩過擬合
# - 使用更大的等效 batch size 穩定梯度
# - 期望 Val Loss 和 Val Acc 的最佳 epoch 更接近

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1219/train.py \
    --exp_name exp59_improved \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1223/runs/exp59_improved \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 300 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 15 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1223/exp59.log

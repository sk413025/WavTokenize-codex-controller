#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# exp_1128 實驗 1: LoRA Rank 32 + Distance Loss 0.05
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 基於 exp_1126/1126-1 的分析:
# - Feature Similarity 改善顯著 (MSE -38%, Cosine +72%)
# - Code Distance 改善有限 (-18%)
# - 假設: LoRA 容量不足 + Distance Loss 權重太小
#
# 本實驗改進:
# - LoRA Rank: 16 -> 32 (參數量 ~2x)
# - Distance Loss Weight: 0.01 -> 0.05 (5x)

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1128

export CUDA_VISIBLE_DEVICES=1

nohup python train.py \
    --exp_name lora_r32_dist0.05 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --feature_loss_weight 1.0 \
    --distance_loss_weight 0.05 \
    --vq_loss_weight 0.0 \
    > experiments/lora_r32_dist0.05.log 2>&1 &

echo "Experiment 1 started: LoRA Rank=32, Distance Loss=0.05"
echo "Monitor: tail -f experiments/lora_r32_dist0.05.log"
echo ""
echo "Expected improvements vs baseline (1126-1):"
echo "  - More LoRA parameters for finer feature control"
echo "  - Stronger distance loss for code alignment"

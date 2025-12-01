#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# exp_1201 實驗 1: Soft Distance Loss Baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 解決 exp_1128 的問題:
# - Distance Loss 不可微 (argmax + indexing)
# - 改用 Soft Distance Loss (softmax 保持梯度)
#
# 基本配置:
# - LoRA Rank: 64 (沿用 exp_1128 最佳配置)
# - Soft Distance Loss Weight: 0.1
# - Temperature: 1.0

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201

export CUDA_VISIBLE_DEVICES=1

nohup python train.py \
    --exp_name soft_dist_baseline \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --feature_loss_weight 1.0 \
    --soft_dist_loss_weight 0.1 \
    --vq_loss_weight 0.0 \
    --temperature 1.0 \
    > experiments/soft_dist_baseline.log 2>&1 &

echo "Experiment 1 started: Soft Distance Loss Baseline"
echo "Monitor: tail -f experiments/soft_dist_baseline.log"
echo ""
echo "Key improvements over exp_1128:"
echo "  - Soft Distance Loss is DIFFERENTIABLE!"
echo "  - Gradient can flow back to LoRA parameters"
echo "  - Distance loss should actually improve during training"

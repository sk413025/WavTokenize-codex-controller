#!/bin/bash
# ============================================================
# Exp32: Larger Batch + Keep LoRA + Moderate Regularization
# ============================================================
#
# 目的: 測試更大 batch size，但保持 LoRA 容量
#
# 基於 Exp30 的教訓:
# - Exp30 減小 LoRA rank (64) → Val Acc 下降 (15-17%)
# - 較小 LoRA capacity 不足以學習 noisy→clean 映射
#
# 改動 (vs Exp27 baseline):
# - LoRA rank: 128 (保持！不改)
# - LoRA alpha: 256 (保持！不改)
# - Batch size: 16 (保持！避免 OOM)
# - Learning rate: 2e-5 → 5e-5 (2.5x，適度增加)
# - lora_dropout: 0.1 → 0.15 (略增正則化)
# - weight_decay: 0.01 → 0.03 (略增正則化)
#
# 理論:
# - 更大 batch → 更穩定梯度
# - 適度增加 LR (不是 10x) → 更快收斂但不發散
# - 保持 LoRA 容量 → 維持表達能力
# - 略增正則化 → 對抗可能的 overfitting
#
# 預期:
# - Val Acc >= Exp27 (17-18%)
# - 訓練更穩定
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

echo "============================================================"
echo "Exp32: Larger Batch + Keep LoRA + Moderate Regularization"
echo "============================================================"
echo "Key changes (vs Exp27):"
echo "  - LoRA rank: 128 (KEEP)"
echo "  - LoRA alpha: 256 (KEEP)"
echo "  - Batch size: 16 (KEEP, avoid OOM)"
echo "  - Learning rate: 5e-5 (was 2e-5, 2.5x)"
echo "  - lora_dropout: 0.15 (was 0.1)"
echo "  - weight_decay: 0.03 (was 0.01)"
echo "============================================================"

python train_lora_v3.py \
    --exp_name exp32_larger_batch_keep_lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.15 \
    --weight_decay 0.03 \
    --feature_weight 1.0 \
    --triplet_weight 0.5 \
    --triplet_margin 0.2 \
    --soft_ce_weight 0.0 \
    --dw_weight 0.0 \
    --lr 5e-5 \
    --batch_size 16 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp32.log

echo ""
echo "============================================================"
echo "Exp32 completed at: $(date)"
echo "============================================================"

#!/bin/bash
# ============================================================
# Exp31: Regularization (Anti-Overfitting)
# ============================================================
#
# 目的: 對抗 13-15% 的 train-val gap (overfitting)
#
# 改動 (基於 Exp27 最佳配置):
# - lora_dropout: 0.1 → 0.2 (增加 dropout)
# - weight_decay: 0.01 → 0.05 (增加 L2 正則化)
#
# 理論:
# - 更高 dropout → 防止 LoRA 層過度依賴特定特徵
# - 更高 weight_decay → 限制權重大小，減少過擬合
# - 保持 Exp27 的 triplet 配置 (已證明有效)
#
# 預期:
# - Train acc 可能略低
# - Val acc 應該更接近 Train acc (縮小 gap)
# - 最終 Val acc 可能更高
# ============================================================

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

echo "============================================================"
echo "Exp31: Regularization (Anti-Overfitting)"
echo "============================================================"
echo "Key changes (vs Exp27):"
echo "  - lora_dropout: 0.2 (was 0.1)"
echo "  - weight_decay: 0.05 (was 0.01)"
echo "============================================================"

python train_lora_v3.py \
    --exp_name exp31_regularized \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --weight_decay 0.05 \
    --feature_weight 1.0 \
    --triplet_weight 0.5 \
    --triplet_margin 0.2 \
    --soft_ce_weight 0.0 \
    --dw_weight 0.0 \
    --lr 2e-5 \
    --batch_size 16 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp31.log

echo ""
echo "============================================================"
echo "Exp31 completed at: $(date)"
echo "============================================================"

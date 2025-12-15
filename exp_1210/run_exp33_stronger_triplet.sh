#!/bin/bash
# ============================================================
# Exp33: Stronger Triplet + Regularization
# ============================================================
#
# 目的: 結合 Exp31 的正則化 + 更強的 Triplet Loss
#
# 基於分析結果:
# - Exp31 (正則化): 18.46% Val Acc - 最佳
# - Exp27 (Feature+Triplet): 18.26% Val Acc
# - Triplet Loss 是關鍵 (+5% over baseline)
# - Soft CE 有害，不使用
#
# 改動 (基於 Exp31):
# - triplet_weight: 0.5 → 0.7 (強化對比學習)
# - triplet_margin: 0.2 → 0.3 (增加分離度)
# - 保持 Exp31 正則化: dropout=0.2, weight_decay=0.05
#
# 理論:
# - 更高 triplet_weight → 更強的對比學習信號
# - 更大 margin → 強制 student 與正確 code 更近，錯誤 code 更遠
# - 結合正則化 → 防止過擬合
#
# 預期:
# - Val Acc > 18.5% (超越 Exp31)
# - 更好的 token 分離度
# ============================================================

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

echo "============================================================"
echo "Exp33: Stronger Triplet + Regularization"
echo "============================================================"
echo "Key changes (vs Exp31):"
echo "  - triplet_weight: 0.7 (was 0.5)"
echo "  - triplet_margin: 0.3 (was 0.2)"
echo "  - lora_dropout: 0.2 (same as Exp31)"
echo "  - weight_decay: 0.05 (same as Exp31)"
echo "============================================================"

python train_lora_v3.py \
    --exp_name exp33_stronger_triplet \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --weight_decay 0.05 \
    --feature_weight 1.0 \
    --triplet_weight 0.7 \
    --triplet_margin 0.3 \
    --soft_ce_weight 0.0 \
    --dw_weight 0.0 \
    --lr 2e-5 \
    --batch_size 16 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp33.log

echo ""
echo "============================================================"
echo "Exp33 completed at: $(date)"
echo "============================================================"

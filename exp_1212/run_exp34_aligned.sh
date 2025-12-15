#!/bin/bash
# ============================================================
# Exp34: Aligned Dataset + Masked Loss
# ============================================================
#
# 修復兩個對齊問題:
# 1. 來源1: Per-pair mismatch → min-length truncation
# 2. 來源2: Cross-sample mismatch → masked loss
#
# 驗證結果 (exp_1212/verify_alignment_issues.py):
# - TRAIN per-pair: 幾乎沒問題 (只差 1 sample)
# - VAL per-pair: 40% 有顯著不一致 (clean 平均長 266ms)
# - Padding 比例: TRAIN ~20%, VAL ~25%
#
# 預期改進:
# - Masked Accuracy 會比 Unmasked Accuracy 更真實
# - 解決 VAL per-pair mismatch 導致的評估偏差
# - 避免 padding frames 稀釋 loss signal
#
# 配置 (與 Exp31 相同，作為 baseline):
# - Feature weight: 1.0
# - Triplet weight: 0.5
# - Triplet margin: 0.2
# - LoRA rank: 128, alpha: 256
# - dropout: 0.2, weight_decay: 0.05
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1212

echo "============================================================"
echo "Exp34: Aligned Dataset + Masked Loss"
echo "============================================================"
echo "修復內容:"
echo "  1. Per-pair min-length truncation (修復來源1)"
echo "  2. Masked loss (修復來源2)"
echo ""
echo "配置 (與 Exp31 相同):"
echo "  - feature_weight: 1.0"
echo "  - triplet_weight: 0.5"
echo "  - triplet_margin: 0.2"
echo "  - lora_dropout: 0.2"
echo "  - weight_decay: 0.05"
echo "============================================================"

python train_aligned.py \
    --exp_name exp34_aligned_masked \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --weight_decay 0.05 \
    --feature_weight 1.0 \
    --triplet_weight 0.5 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 2e-5 \
    --batch_size 16 \
    --num_epochs 1000 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp34.log

echo ""
echo "============================================================"
echo "Exp34 completed at: $(date)"
echo "============================================================"

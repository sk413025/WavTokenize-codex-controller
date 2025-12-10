#!/bin/bash
# ============================================================
# Exp26: Feature Only (修復版基線)
# ============================================================
#
# 目的: 驗證 codebook 漂移修復後，Feature Loss 是否能正常工作
#
# 修復內容:
# 1. Teacher 始終保持 eval 模式
# 2. Teacher/Student quantizer 都凍結
# 3. Codebook 安全檢查
#
# Loss 配置:
# - Feature MSE: 1.0 (唯一優化目標)
# - Triplet: 0.0 (關閉)
# - DW: 0.0 (關閉)
#
# 預期:
# - Feature Loss 下降 → VQ Distance 下降 → Token Acc 上升
# - 這是驗證修復成功的關鍵實驗
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

echo "============================================================"
echo "Exp26: Feature Only (Fixed Codebook Baseline)"
echo "============================================================"
echo "Fixes applied:"
echo "  - Teacher always in eval mode"
echo "  - All quantizers frozen (no EMA update)"
echo "  - Codebook integrity check every 100 batches"
echo "============================================================"

python train_lora_v3.py \
    --exp_name exp26_feature_only_fixed \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_weight 1.0 \
    --triplet_weight 0.0 \
    --triplet_margin 0.2 \
    --dw_weight 0.0 \
    --soft_ce_weight 0.0 \
    --lr 2e-5 \
    --batch_size 8 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp26.log

echo ""
echo "============================================================"
echo "Exp26 completed at: $(date)"
echo "============================================================"

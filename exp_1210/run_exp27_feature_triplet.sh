#!/bin/bash
# ============================================================
# Exp27: Feature + Triplet (修復版)
# ============================================================
#
# 目的: 驗證 Triplet Loss 是否能進一步提升效果
#
# Loss 配置:
# - Feature MSE: 1.0
# - Triplet: 0.5 (與 Exp26 對比)
# - DW: 0.0 (關閉)
#
# 與 Exp23 的差異:
# - Exp23 有 codebook 漂移問題
# - Exp27 已修復，結果更可靠
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

echo "============================================================"
echo "Exp27: Feature + Triplet (Fixed)"
echo "============================================================"

python train_lora_v3.py \
    --exp_name exp27_feature_triplet_fixed \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_weight 1.0 \
    --triplet_weight 0.5 \
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
    2>&1 | tee exp27.log

echo ""
echo "============================================================"
echo "Exp27 completed at: $(date)"
echo "============================================================"

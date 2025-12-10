#!/bin/bash
# ============================================================
# Exp20: Adapter + Triplet Loss + CE Loss
# ============================================================
#
# 方案 D 完整版: Adapter + 三種 Loss
# Loss: Feature MSE + Triplet Loss + Cross-Entropy Loss
#
# 預期結果: 15-25% Token Accuracy
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1209

echo "============================================================"
echo "Exp20: Adapter + Triplet + CE Loss"
echo "============================================================"
echo "Start time: $(date)"
echo ""

python train_adapter.py \
    --exp_name exp20_adapter_triplet_ce \
    --adapter_hidden 256 \
    --adapter_layers 2 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.5 \
    --ce_weight 0.1 \
    --ce_temperature 0.1 \
    --lr 1e-4 \
    --batch_size 8 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    2>&1 | tee exp20.log

echo ""
echo "============================================================"
echo "Exp20 completed at: $(date)"
echo "============================================================"

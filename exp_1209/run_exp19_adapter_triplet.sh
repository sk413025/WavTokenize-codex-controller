#!/bin/bash
# ============================================================
# Exp19: Adapter + Triplet Loss
# ============================================================
#
# 方案 D: 使用 DenoiseAdapter 修正 encoder 輸出
# Loss: Feature MSE + Triplet Loss
#
# 預期結果: 10-20% Token Accuracy
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1209

echo "============================================================"
echo "Exp19: Adapter + Triplet Loss"
echo "============================================================"
echo "Start time: $(date)"
echo ""

python train_adapter.py \
    --exp_name exp19_adapter_triplet \
    --adapter_hidden 256 \
    --adapter_layers 2 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.5 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --batch_size 28 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    2>&1 | tee exp19.log

echo ""
echo "============================================================"
echo "Exp19 completed at: $(date)"
echo "============================================================"

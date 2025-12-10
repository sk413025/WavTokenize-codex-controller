#!/bin/bash
# ============================================================
# Exp21: 擴大 LoRA + Triplet Loss
# ============================================================
#
# 方案 B+A: 18 層 LoRA (rank=256) + Triplet Loss
# Loss: Feature MSE + Triplet Loss
#
# 可訓練參數: ~3.7M (4.4%)
# 預期結果: 20-35% Token Accuracy
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1209

echo "============================================================"
echo "Exp21: Expanded LoRA (18 layers, rank=256) + Triplet Loss"
echo "============================================================"
echo "Start time: $(date)"
echo ""

python train_lora_expanded.py \
    --exp_name exp21_lora_triplet \
    --lora_rank 256 \
    --lora_alpha 512 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.5 \
    --ce_weight 0.0 \
    --lr 5e-5 \
    --batch_size 8 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    2>&1 | tee exp21.log

echo ""
echo "============================================================"
echo "Exp21 completed at: $(date)"
echo "============================================================"

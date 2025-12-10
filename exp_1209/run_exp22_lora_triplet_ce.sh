#!/bin/bash
# ============================================================
# Exp22: 擴大 LoRA + Triplet Loss + CE Loss
# ============================================================
#
# 方案 B+A 完整版: 18 層 LoRA + 三種 Loss
# Loss: Feature MSE + Triplet Loss + Cross-Entropy Loss
#
# 可訓練參數: ~3.7M (4.4%)
# 預期結果: 25-40% Token Accuracy
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1209

echo "============================================================"
echo "Exp22: Expanded LoRA + Triplet + CE Loss"
echo "============================================================"
echo "Start time: $(date)"
echo ""

python train_lora_expanded.py \
    --exp_name exp22_lora_triplet_ce \
    --lora_rank 256 \
    --lora_alpha 512 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.5 \
    --ce_weight 0.1 \
    --ce_temperature 0.1 \
    --lr 5e-5 \
    --batch_size 8 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    2>&1 | tee exp22.log

echo ""
echo "============================================================"
echo "Exp22 completed at: $(date)"
echo "============================================================"

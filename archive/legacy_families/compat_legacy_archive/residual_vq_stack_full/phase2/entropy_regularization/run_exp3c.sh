#!/bin/bash

# ============================================================
# exp_0128 Phase 2: 實驗 3c - Entropy Regularization (λ=0.10)
# ============================================================

set -e

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 切換到工作目錄
cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 環境變數設定
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 實驗參數
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="exp_0128/phase2/entropy_regularization/run_exp3c_${TIMESTAMP}"
LAMBDA_ENTROPY=0.10

echo "============================================================"
echo "Starting Experiment 3c: Entropy Regularization (λ=0.10)"
echo "============================================================"
echo "GPU: 0"
echo "Lambda Entropy: ${LAMBDA_ENTROPY}"
echo "Output: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "============================================================"

# 執行訓練 (與 exp_k v6 baseline 一致: batch_size=8, grad_accum=2, effective=16)
PYTHONUNBUFFERED=1 python exp_0128/phase2/entropy_regularization/train_short_run.py \
    --steps 1000 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lambda_entropy ${LAMBDA_ENTROPY} \
    --output_dir ${OUTPUT_DIR} \
    --seed 42 \
    --device cuda:0 \
    --eval_interval 200 \
    |& tee ${OUTPUT_DIR}.log

echo ""
echo "============================================================"
echo "Experiment 3c Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"

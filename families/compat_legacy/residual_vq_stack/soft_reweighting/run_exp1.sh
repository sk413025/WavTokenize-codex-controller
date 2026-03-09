#!/bin/bash

# ============================================================
# exp_0128: 實驗 1 - TracIn-Weighted Soft Reweighting (GPU 1)
# ============================================================

set -e

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 切換到工作目錄
cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 環境變數設定
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 實驗參數
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="exp_0128/soft_reweighting/run_exp1_${TIMESTAMP}"
ALPHA=0.5  # Reweighting strength (可調整: 0.3, 0.5, 0.7)

echo "============================================================"
echo "Starting Experiment 1: TracIn-Weighted Soft Reweighting"
echo "============================================================"
echo "GPU: 1"
echo "Alpha: ${ALPHA}"
echo "Output: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "============================================================"

# 執行訓練 (與 exp_k v6 baseline 一致: batch_size=8, grad_accum=2, effective=16)
PYTHONUNBUFFERED=1 python exp_0128/soft_reweighting/train_short_run.py \
    --steps 1000 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 1e-4 \
    --alpha ${ALPHA} \
    --tracin_scores_csv exp_0125/tracin_token_collapse_589e6d/tracin_scores_5ckpt.csv \
    --output_dir ${OUTPUT_DIR} \
    --seed 42 \
    --device cuda:0 \
    --eval_interval 200 \
    |& tee ${OUTPUT_DIR}.log

echo ""
echo "============================================================"
echo "Experiment 1 Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"

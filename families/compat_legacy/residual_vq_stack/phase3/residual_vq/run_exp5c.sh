#!/bin/bash

# ============================================================
# exp_0128 Phase 3: 實驗 5c - RVQ (8 layers, 512 codes/layer)
# ============================================================

set -e
set -o pipefail

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
OUTPUT_DIR="exp_0128/phase3/residual_vq/run_exp5c_${TIMESTAMP}"
N_LAYERS=8
CODEBOOK_SIZE=512

echo "============================================================"
echo "Starting Experiment 5c: RVQ (8 layers, 512 codes/layer)"
echo "============================================================"
echo "GPU: 1"
echo "RVQ Layers: ${N_LAYERS}"
echo "Codebook Size per Layer: ${CODEBOOK_SIZE}"
echo "Total Expressiveness: ${CODEBOOK_SIZE}^${N_LAYERS} = 5.3e21"
echo "Output: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "============================================================"

# 執行訓練 (與 baseline 一致: batch_size=8, grad_accum=2, effective=16)
PYTHONUNBUFFERED=1 python exp_0128/phase3/residual_vq/train_rvq_short_run.py \
    --steps 1000 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 1e-4 \
    --n_rvq_layers ${N_LAYERS} \
    --rvq_codebook_size ${CODEBOOK_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    --seed 42 \
    --device cuda:0 \
    --eval_interval 200 \
    |& tee ${OUTPUT_DIR}.log

echo ""
echo "============================================================"
echo "Experiment 5c Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"

#!/bin/bash
#
# exp_0128: Noise-Balanced Sampling Short-Run (實驗 2)
#
# 目的：驗證平衡噪音材質是否能改善 token collapse
#
# 執行：
#     bash exp_0128/noise_balanced_sampling/run_exp2.sh
#

set -e

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 切換到工作目錄
cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 環境變數設定
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 配置 (與 exp_k v6 baseline 一致)
GPU=0
STEPS=1000
BATCH_SIZE=8
GRAD_ACCUM=2
LR=1e-4
SEED=42

# 輸出目錄
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="exp_0128/noise_balanced_sampling/run_exp2_${TIMESTAMP}"

echo "=================================================="
echo "Exp 0128: Noise-Balanced Sampling (實驗 2)"
echo "=================================================="
echo "GPU: ${GPU}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accum: ${GRAD_ACCUM}"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Learning rate: ${LR}"
echo "Seed: ${SEED}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=================================================="

# Run training
PYTHONUNBUFFERED=1 python \
    exp_0128/noise_balanced_sampling/train_short_run.py \
    --steps ${STEPS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --eval_interval 200 \
    |& tee ${OUTPUT_DIR}.log

echo ""
echo "✅ Experiment complete!"
echo "Results: ${OUTPUT_DIR}/summary.json"
echo "Log: ${OUTPUT_DIR}.log"

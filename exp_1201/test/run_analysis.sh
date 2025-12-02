#!/bin/bash
# exp_1201/test/run_analysis.sh
#
# 運行特徵和 Token 分析腳本
#
# 使用方式:
#   bash run_analysis.sh ste_baseline
#   bash run_analysis.sh ce_balanced
#   bash run_analysis.sh gumbel_baseline

set -e

EXP_NAME=${1:-"ste_baseline"}
NUM_SAMPLES=${2:-30}
DEVICE=${3:-"cuda:0"}

echo "========================================"
echo "Feature & Token Analysis for: ${EXP_NAME}"
echo "Number of samples: ${NUM_SAMPLES}"
echo "Device: ${DEVICE}"
echo "========================================"

cd "$(dirname "$0")"

# 1. t-SNE 特徵空間分析
echo ""
echo "=========================================="
echo "Running t-SNE Feature Analysis..."
echo "=========================================="
python feature_tsne_analysis.py \
    --exp_name ${EXP_NAME} \
    --num_samples ${NUM_SAMPLES} \
    --device ${DEVICE}

# 2. Token 距離分析
echo ""
echo "=========================================="
echo "Running Token Distance Analysis..."
echo "=========================================="
python token_distance_analysis.py \
    --exp_name ${EXP_NAME} \
    --num_samples ${NUM_SAMPLES} \
    --device ${DEVICE}

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "Results saved to: ./results/${EXP_NAME}/"
echo "=========================================="
ls -la ./results/${EXP_NAME}/

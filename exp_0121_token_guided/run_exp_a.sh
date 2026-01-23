#!/bin/bash
# ============================================================
# Exp A: 標準 Exp K 架構 (Baseline)
#
# 核心:
#   沿用 Exp K 驗證有效的架構
#   - MSE + Triplet + 中間層監督 (L3, L4, L6)
#   - 全層 LoRA (18 層)
#   - Curriculum Learning
#
# 目的:
#   建立 baseline，後續實驗與此比較
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 確保 runs 目錄存在
mkdir -p runs

EXP_NAME="exp_a_baseline_$(date +%Y%m%d_%H%M%S)"
echo "Starting experiment: $EXP_NAME"
echo "  Architecture: Exp K (MSE + Triplet + Intermediate)"
echo "  LoRA: 18 layers, rank=256"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name "$EXP_NAME" \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --intermediate_weight 0.5 \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --num_epochs 300 \
    --curriculum_start 0.3 \
    --curriculum_end 0.85 \
    --curriculum_epochs 200 \
    --save_audio_interval 50 \
    2>&1 | tee "runs/${EXP_NAME}.log"

echo "Experiment $EXP_NAME completed!"

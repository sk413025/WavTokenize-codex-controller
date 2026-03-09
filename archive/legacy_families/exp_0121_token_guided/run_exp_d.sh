#!/bin/bash
# ============================================================
# Exp D: 平衡 Token Acc 和 STOI
#
# 策略:
#   1. 降低 high_error_multiplier: 2.0 → 1.5
#   2. 加入 STOI 正則化 (通過降低 token_weighted 權重)
#
# 目標:
#   - Val Acc > 0.95% (接近 Exp B)
#   - STOI > 0.35 (比 Exp B 的 0.227 好)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p runs

# 確保分析輸出存在
if [ ! -f "analysis_outputs/token_error_rates.pt" ]; then
    echo "Running token error analysis..."
    python analyze_error_tokens.py
fi

EXP_NAME="exp_d_balanced_$(date +%Y%m%d_%H%M%S)"
echo "Starting experiment: $EXP_NAME"
echo "  Strategy: Lower multiplier (1.5) for better STOI"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name "$EXP_NAME" \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --intermediate_weight 0.5 \
    --use_token_weighted \
    --token_error_rates_path analysis_outputs/token_error_rates.pt \
    --high_error_threshold 0.7 \
    --high_error_multiplier 1.5 \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --num_epochs 300 \
    --curriculum_start 0.3 \
    --curriculum_end 0.85 \
    --curriculum_epochs 200 \
    --save_audio_interval 50 \
    2>&1 | tee "runs/${EXP_NAME}.log"

echo "Experiment $EXP_NAME completed!"

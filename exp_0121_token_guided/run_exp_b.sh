#!/bin/bash
# ============================================================
# Exp B: Exp K + Token-Weighted Loss
#
# 核心:
#   在 Exp K 架構上加入 Token-Weighted Loss
#   - 對高錯誤率 token (>0.7) 給 2x 權重
#
# 基於分析結果:
#   - 許多 token error_rate > 0.8
#   - 這些 token 應該被優先修正
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 確保 runs 目錄存在
mkdir -p runs

# 1. 先執行 token 分析 (如果還沒有)
if [ ! -f "analysis_outputs/token_error_rates.pt" ]; then
    echo "Running token error analysis..."
    python analyze_error_tokens.py
fi

# 2. 執行訓練
EXP_NAME="exp_b_token_weighted_$(date +%Y%m%d_%H%M%S)"
echo "Starting experiment: $EXP_NAME"
echo "  Architecture: Exp K + Token-Weighted Loss"
echo "  LoRA: 18 layers, rank=256"

CUDA_VISIBLE_DEVICES=1 python train.py \
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
    --high_error_multiplier 2.0 \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --num_epochs 300 \
    --curriculum_start 0.3 \
    --curriculum_end 0.85 \
    --curriculum_epochs 200 \
    --save_audio_interval 50 \
    2>&1 | tee "runs/${EXP_NAME}.log"

echo "Experiment $EXP_NAME completed!"

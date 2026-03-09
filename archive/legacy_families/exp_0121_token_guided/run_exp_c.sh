#!/bin/bash
# ============================================================
# Exp C: Layer-Selective LoRA
#
# 核心:
#   只在「噪音敏感層」加 LoRA，減少參數量同時針對問題
#   沿用 Exp K 的 Loss 架構
#
# 基於 exp_1231_feature 分析:
#   - model[4] ResBlock2: 噪音敏感度 0.80 (最高)
#   - model[6] Downsample2: 噪音敏感度 0.79
#
# 對應 18 層 conv 名稱 (只選 3 層):
#   - L5: model.4.block.1
#   - L6: model.4.block.3
#   - L8: model.6.conv
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 確保 runs 目錄存在
mkdir -p runs

EXP_NAME="exp_c_layer_selective_$(date +%Y%m%d_%H%M%S)"
echo "Starting experiment: $EXP_NAME"
echo "  Architecture: Exp K (MSE + Triplet + Intermediate)"
echo "  LoRA: 3 noise-sensitive layers only, rank=128"

# --lora_target_layers 指定只在特定層加 LoRA
# 使用較大的 rank 因為層數少
CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name "$EXP_NAME" \
    --lora_target_layers "4.block.1" "4.block.3" "6.conv" \
    --lora_rank 128 \
    --lora_alpha 256 \
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

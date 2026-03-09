#!/bin/bash
# ============================================================
# Exp E: Curriculum Token Weighting
#
# 策略:
#   - 前期 (epoch 1-100): 不使用 token weighting (保護 STOI)
#   - 後期 (epoch 100-300): 漸進加入 token weighting
#
# 原理:
#   - Exp B 的 STOI 在 epoch 50 急劇下降 (0.421→0.150)
#   - 讓模型先學好基礎音質，再微調 token accuracy
#
# 實現:
#   - Phase 1 (1-100): multiplier = 1.0 (等同無加權)
#   - Phase 2 (100-300): 使用 Exp B 配置
#
# 這需要修改 train.py 支援 curriculum token weighting
# 暫時用 Exp A (無 token weighting) 訓練 100 epochs
# 然後接續用 Exp B 配置
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p runs

EXP_NAME="exp_e_curriculum_token_$(date +%Y%m%d_%H%M%S)"
echo "Starting experiment: $EXP_NAME"
echo "  Strategy: Two-phase training"
echo "  Phase 1: No token weighting (protect STOI)"
echo "  Phase 2: Token weighting (boost Acc)"

# Phase 1: 基礎訓練 (無 token weighting)
echo ""
echo "============================================================"
echo "Phase 1: Basic training without token weighting (100 epochs)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name "${EXP_NAME}_phase1" \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --intermediate_weight 0.5 \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --num_epochs 100 \
    --curriculum_start 0.3 \
    --curriculum_end 0.85 \
    --curriculum_epochs 100 \
    --save_audio_interval 50 \
    2>&1 | tee "runs/${EXP_NAME}_phase1.log"

echo "Phase 1 completed!"

# 找到 Phase 1 的 checkpoint
PHASE1_DIR=$(ls -td runs/${EXP_NAME}_phase1_*/ 2>/dev/null | head -1)
if [ -z "$PHASE1_DIR" ]; then
    echo "Error: Phase 1 directory not found!"
    exit 1
fi

PHASE1_CKPT="${PHASE1_DIR}best_model.pt"
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "Error: Phase 1 checkpoint not found at $PHASE1_CKPT"
    exit 1
fi

echo ""
echo "============================================================"
echo "Phase 2: Fine-tune with token weighting (200 epochs)"
echo "  Loading from: $PHASE1_CKPT"
echo "============================================================"

# Phase 2: Token-weighted fine-tuning
# 注意: 需要 train.py 支援 --resume_from 參數
# 暫時跳過，建議先測試 Exp D

echo "Phase 2 requires --resume_from support in train.py"
echo "Please run Exp D first as a simpler alternative."

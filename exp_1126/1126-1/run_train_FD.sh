#!/bin/bash

# LoRA Encoder Denoising Training Script - 純 Feature Distillation 版本
# 策略 B：只使用 Feature Loss，不使用 Distance Loss
# 基於 done/exp/lora_encoder_denoising

set -e

# 使用 GPU 1 (可根據需要修改)
export CUDA_VISIBLE_DEVICES=1

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="lora_encoder_1126_1_FD"  # FD = Feature Distillation

# 訓練超參數
BATCH_SIZE=16
NUM_EPOCHS=50
LEARNING_RATE=5e-5
LORA_RANK=16
LORA_ALPHA=32

# Loss 權重 - 純 Feature Distillation
FEATURE_LOSS_WEIGHT=1.0
DISTANCE_LOSS_WEIGHT=0.0   # 關閉 Distance Loss
VQ_LOSS_WEIGHT=0.0         # 關閉 VQ Loss

echo "========================================================================"
echo "LoRA Encoder Denoising - 純 Feature Distillation (策略 B)"
echo "========================================================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Experiment: $EXP_NAME"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo ""
echo "✓ Loss 權重設定:"
echo "  - Feature Loss Weight: $FEATURE_LOSS_WEIGHT (MSE between features)"
echo "  - Distance Loss Weight: $DISTANCE_LOSS_WEIGHT (關閉)"
echo "  - VQ Loss Weight: $VQ_LOSS_WEIGHT (關閉)"
echo ""
echo "✓ 策略說明:"
echo "  - 只優化 MSE(student_features, teacher_features)"
echo "  - Token Accuracy 僅作為監控指標，不參與優化"
echo "  - 理論：feature 一致 → token 也會一致 (同一個 VQ codebook)"
echo ""

cd "$SCRIPT_DIR"

# 創建必要目錄
mkdir -p experiments checkpoints logs

# 運行訓練 (路徑在 config.py 中設定)
python train.py \
    --exp_name ${EXP_NAME} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --feature_loss_weight ${FEATURE_LOSS_WEIGHT} \
    --distance_loss_weight ${DISTANCE_LOSS_WEIGHT} \
    --vq_loss_weight ${VQ_LOSS_WEIGHT} \
    --num_workers 4 \
    2>&1 | tee experiments/${EXP_NAME}.log

echo ""
echo "========================================================================"
echo "Training completed!"
echo "========================================================================"

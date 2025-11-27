#!/bin/bash

# LoRA Encoder Denoising Training Script
# 重現 Teacher-Student Knowledge Distillation + LoRA Fine-tuning 實驗
# 基於 done/exp/lora_encoder_denoising

set -e

# 使用 GPU 0 (可根據需要修改)
export CUDA_VISIBLE_DEVICES=1

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="lora_encoder_1126_1"

# 數據路徑
TRAIN_CACHE="/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt"
VAL_CACHE="/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt"

# WavTokenizer 配置
WAVTOK_CONFIG="/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT="/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

# 訓練超參數
BATCH_SIZE=16
NUM_EPOCHS=50
LEARNING_RATE=5e-5
LORA_RANK=16
LORA_ALPHA=32

echo "========================================================================"
echo "LoRA Encoder Denoising - Teacher-Student 架構"
echo "========================================================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Experiment: $EXP_NAME"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo ""
echo "✓ 架構特點:"
echo "  - Teacher: 凍結的原始 WavTokenizer Encoder"
echo "  - Student: Encoder + LoRA (僅訓練 ~19K 參數)"
echo "  - Loss: Feature MSE + Distance Loss + VQ Loss"
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
    --num_workers 4 \
    2>&1 | tee experiments/${EXP_NAME}.log

echo ""
echo "========================================================================"
echo "Training completed!"
echo "========================================================================"

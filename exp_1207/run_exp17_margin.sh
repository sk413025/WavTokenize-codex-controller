#!/bin/bash

# exp17: Margin-based Contrastive Loss
# 使用 Margin Loss 替換 MSE Loss，直接優化 Voronoi boundary

# 實驗設定
EXP_NAME="exp17_margin_loss"
EPOCHS=30
BATCH_SIZE=4
LR=5e-5

# Loss weights
MARGIN=0.5        # Margin 大小
CE_WEIGHT=1.0     # CE Loss 權重

# LoRA settings
LORA_RANK=64
LORA_ALPHA=128

# Logging
LOG_FILE="exp17.log"

echo "========================================" | tee -a $LOG_FILE
echo "實驗: $EXP_NAME" | tee -a $LOG_FILE
echo "開始時間: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "實驗設定:" | tee -a $LOG_FILE
echo "  - Epochs: $EPOCHS" | tee -a $LOG_FILE
echo "  - Batch Size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "  - Learning Rate: $LR" | tee -a $LOG_FILE
echo "  - Margin: $MARGIN" | tee -a $LOG_FILE
echo "  - CE Weight: $CE_WEIGHT" | tee -a $LOG_FILE
echo "  - LoRA Rank: $LORA_RANK" | tee -a $LOG_FILE
echo "  - LoRA Alpha: $LORA_ALPHA" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 運行訓練
CUDA_VISIBLE_DEVICES=0 python train_margin_loss.py \
    --exp_name $EXP_NAME \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --margin $MARGIN \
    --ce_weight $CE_WEIGHT \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --log_interval 50 \
    --save_interval 10 \
    --plot_interval 10 \
    --audio_interval 10 \
    --num_audio_samples 3 \
    2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "實驗完成!" | tee -a $LOG_FILE
echo "結束時間: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

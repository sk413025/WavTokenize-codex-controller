#!/bin/bash

# exp18: 反向 Curriculum Learning (CE first → CE+MSE → MSE dominant)
# Stage 1: 只用 CE，快速定位正確 Voronoi cell
# Stage 2: CE + MSE，在 cell 內部精細化
# Stage 3: MSE dominant，穩定 embedding
export CUDA_VISIBLE_DEVICES=1
# 實驗設定
EXP_NAME="exp18_reverse_curriculum"
TOTAL_EPOCHS=50
BATCH_SIZE=4
LR=5e-5

# Stage 1 (epoch 1-10): CE only
STAGE1_EPOCHS=10
STAGE1_CE_WEIGHT=1.0
STAGE1_MSE_WEIGHT=0.0

# Stage 2 (epoch 11-30): CE + MSE
STAGE2_EPOCHS=20
STAGE2_CE_WEIGHT=0.5
STAGE2_MSE_WEIGHT=1.0

# Stage 3 (epoch 31-50): MSE dominant
STAGE3_EPOCHS=20
STAGE3_CE_WEIGHT=0.1
STAGE3_MSE_WEIGHT=1.0

# LoRA settings
LORA_RANK=64
LORA_ALPHA=128

# Logging
LOG_FILE="exp18.log"

echo "========================================" | tee -a $LOG_FILE
echo "實驗: $EXP_NAME" | tee -a $LOG_FILE
echo "開始時間: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "實驗設定:" | tee -a $LOG_FILE
echo "  - Total Epochs: $TOTAL_EPOCHS" | tee -a $LOG_FILE
echo "  - Batch Size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "  - Learning Rate: $LR" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "  Stage 1 (epoch 1-$STAGE1_EPOCHS): CE only" | tee -a $LOG_FILE
echo "    - CE Weight: $STAGE1_CE_WEIGHT" | tee -a $LOG_FILE
echo "    - MSE Weight: $STAGE1_MSE_WEIGHT" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "  Stage 2 (epoch $((STAGE1_EPOCHS+1))-$((STAGE1_EPOCHS+STAGE2_EPOCHS))): CE + MSE" | tee -a $LOG_FILE
echo "    - CE Weight: $STAGE2_CE_WEIGHT" | tee -a $LOG_FILE
echo "    - MSE Weight: $STAGE2_MSE_WEIGHT" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "  Stage 3 (epoch $((STAGE1_EPOCHS+STAGE2_EPOCHS+1))-$TOTAL_EPOCHS): MSE dominant" | tee -a $LOG_FILE
echo "    - CE Weight: $STAGE3_CE_WEIGHT" | tee -a $LOG_FILE
echo "    - MSE Weight: $STAGE3_MSE_WEIGHT" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "  - LoRA Rank: $LORA_RANK" | tee -a $LOG_FILE
echo "  - LoRA Alpha: $LORA_ALPHA" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 運行訓練
python train_curriculum.py \
    --exp_name $EXP_NAME \
    --num_epochs $TOTAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --stage1_epochs $STAGE1_EPOCHS \
    --stage1_ce_weight $STAGE1_CE_WEIGHT \
    --stage1_mse_weight $STAGE1_MSE_WEIGHT \
    --stage2_epochs $STAGE2_EPOCHS \
    --stage2_ce_weight $STAGE2_CE_WEIGHT \
    --stage2_mse_weight $STAGE2_MSE_WEIGHT \
    --stage3_epochs $STAGE3_EPOCHS \
    --stage3_ce_weight $STAGE3_CE_WEIGHT \
    --stage3_mse_weight $STAGE3_MSE_WEIGHT \
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

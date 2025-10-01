#!/bin/bash

# CrossEntropy實驗 - 從檢查點繼續訓練
# 從epoch 70繼續完成200個epoch的訓練

# 設置實驗參數
EXP_ID="202509300125_resumed"
BATCH_SIZE=8
NUM_WORKERS=4
OUTPUT_DIR="results/crossentropy_exp_${EXP_ID}"
LOG_FILE="logs/crossentropy_experiment_${EXP_ID}.log"

# 設置檢查點路徑 - 使用最新的檢查點
RESUME_CHECKPOINT="results/crossentropy_exp_202509300125_resumed/model_epoch_75.pth"

# 環境變量設置
export TTT_BATCH_SIZE="$BATCH_SIZE"
export TTT_NUM_WORKERS="$NUM_WORKERS" 
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CONTENT_BATCHING=true

# GPU 配置 - 使用GPU 0 (正常的RTX 2080 Ti)
export CUDA_VISIBLE_DEVICES=0

# 創建必要目錄
mkdir -p logs
mkdir -p results
mkdir -p "$OUTPUT_DIR"

echo "====================================================="
echo "CrossEntropy實驗恢復 - 從檢查點繼續訓練"
echo "====================================================="
echo "- 實驗ID: $EXP_ID"
echo "- 恢復檢查點: $RESUME_CHECKPOINT" 
echo "- 從epoch 76開始，目標: epoch 200"
echo "- GPU使用: GPU 0 (RTX 2080 Ti)"
echo "- 批次大小: $BATCH_SIZE"
echo "- 輸出目錄: $OUTPUT_DIR"
echo "- 日誌文件: $LOG_FILE"
echo "====================================================="

# 檢查檢查點檔案是否存在
if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "❌ 錯誤: 檢查點檔案不存在: $RESUME_CHECKPOINT"
    exit 1
fi

echo "✅ 檢查點檔案存在，準備恢復訓練..."

# 激活conda環境
echo "激活conda test環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 清理CUDA緩存
echo "清理CUDA緩存..."
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 執行恢復訓練
echo "🚀 從epoch 70恢復CrossEntropy實驗，時間: $(date)"

python wavtokenizer_transformer_denoising.py \
    --d_model 128 \
    --nhead 2 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --dim_feedforward 256 \
    --max_length 256 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 25 \
    --val_speakers girl9 boy7 \
    --train_speakers boy1 boy3 boy4 boy5 boy6 girl2 girl3 girl4 girl6 girl7 \
    --max_sentences_per_speaker 100 \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    2>&1 | tee -a $LOG_FILE

echo ""
echo "====================================================="
echo "CrossEntropy實驗恢復完成，時間: $(date)"
echo "結果日誌保存在: $LOG_FILE"
echo "實驗結果保存在: $OUTPUT_DIR"
echo "====================================================="
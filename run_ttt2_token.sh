#!/bin/bash

# TTT2 Token Enhancement 實驗執行腳本
# 日期: $(date +%Y%m%d_%H%M%S)
# 目的: 執行 Token-based Feature Enhancement 訓練

echo "============================================"
echo "TTT2 Token Enhancement 實驗"
echo "實驗時間: $(date)"
echo "============================================"

# 檢查 GPU 可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo "錯誤: 找不到 nvidia-smi，請確認 CUDA 已安裝"
    exit 1
fi

echo ""
echo "GPU 狀態:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo ""

# 自動選擇空閒的 GPU (排除 GPU 2)
echo "選擇最空閒的 GPU (排除 GPU 2)..."
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
BEST_GPU=$(echo "$GPU_INFO" | awk -F',' '$1 == 0 || $1 == 1 {print $1, $2, $3}' | sort -k2,2n | head -1 | awk '{print $1}')

if [ -z "$BEST_GPU" ]; then
    echo "錯誤: 無法找到可用的 GPU (GPU 0 或 GPU 1)"
    exit 1
fi

echo "選擇的 GPU: $BEST_GPU"
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# 實驗參數
BATCH_SIZE=8
NUM_EPOCHS=100
LEARNING_RATE=1e-4
EMBED_DIM=512
ENHANCER_LAYERS=4
ENHANCER_HEADS=8
ENHANCER_FF_DIM=2048
DROPOUT=0.1

# 損失權重
LOSS_WEIGHT_TOKEN_CE=0.4
LOSS_WEIGHT_FEATURE_L2=0.3
LOSS_WEIGHT_AUDIO_L1=0.2
LOSS_WEIGHT_TOKEN_SMOOTH=0.1

# 數據參數
TRAIN_SPEAKERS="boy1 boy3 boy4 boy5 boy6 girl2 girl3 girl4 girl6 girl7"
VAL_SPEAKERS="girl9 boy7"
MAX_SENTENCES_PER_SPEAKER=""  # 留空表示使用全部句子

# 模型路徑
CONFIG_PATH="./config/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
MODEL_PATH="/home/sbplab/ruizi/WavTokenize/wavtokenizer_large_speech_320_24k.ckpt"

# 輸出目錄
OUTPUT_DIR="./results/ttt2_token_enhancement"

# 日誌檔案
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/ttt2_token_${TIMESTAMP}.log"
mkdir -p ./logs

echo ""
echo "實驗配置:"
echo "  - GPU: $BEST_GPU"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Embed Dim: $EMBED_DIM"
echo "  - Enhancer Layers: $ENHANCER_LAYERS"
echo "  - Enhancer Heads: $ENHANCER_HEADS"
echo "  - Enhancer FF Dim: $ENHANCER_FF_DIM"
echo "  - Dropout: $DROPOUT"
echo ""
echo "損失權重:"
echo "  - Token CE: $LOSS_WEIGHT_TOKEN_CE"
echo "  - Feature L2: $LOSS_WEIGHT_FEATURE_L2"
echo "  - Audio L1: $LOSS_WEIGHT_AUDIO_L1"
echo "  - Token Smooth: $LOSS_WEIGHT_TOKEN_SMOOTH"
echo ""
echo "數據集:"
echo "  - 訓練語者: $TRAIN_SPEAKERS"
echo "  - 驗證語者: $VAL_SPEAKERS"
echo ""
echo "輸出目錄: $OUTPUT_DIR"
echo "日誌檔案: $LOG_FILE"
echo ""
echo "============================================"
echo "開始訓練..."
echo "============================================"
echo ""

# 構建參數字符串
MAX_SENTENCES_ARG=""
if [ -n "$MAX_SENTENCES_PER_SPEAKER" ]; then
    MAX_SENTENCES_ARG="--max_sentences_per_speaker $MAX_SENTENCES_PER_SPEAKER"
fi

# 執行訓練
python ttt2_token.py \
    --config_path "$CONFIG_PATH" \
    --model_path "$MODEL_PATH" \
    --embed_dim $EMBED_DIM \
    --enhancer_layers $ENHANCER_LAYERS \
    --enhancer_heads $ENHANCER_HEADS \
    --enhancer_ff_dim $ENHANCER_FF_DIM \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 1e-5 \
    --num_workers 4 \
    --train_speakers $TRAIN_SPEAKERS \
    --val_speakers $VAL_SPEAKERS \
    $MAX_SENTENCES_ARG \
    --loss_weight_token_ce $LOSS_WEIGHT_TOKEN_CE \
    --loss_weight_feature_l2 $LOSS_WEIGHT_FEATURE_L2 \
    --loss_weight_audio_l1 $LOSS_WEIGHT_AUDIO_L1 \
    --loss_weight_token_smooth $LOSS_WEIGHT_TOKEN_SMOOTH \
    --output_dir "$OUTPUT_DIR" \
    --save_every 10 \
    --seed 42 \
    --device cuda 2>&1 | tee "$LOG_FILE"

# 檢查訓練是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ 訓練成功完成！"
    echo "============================================"
    echo "結果目錄: $OUTPUT_DIR"
    echo "日誌檔案: $LOG_FILE"
    echo ""
    
    # 列出生成的檔案
    echo "生成的檔案:"
    ls -lh "$OUTPUT_DIR"/exp_*/checkpoints/*.pth 2>/dev/null || echo "  (檢查點檔案)"
    ls -lh "$OUTPUT_DIR"/exp_*/*.png 2>/dev/null || echo "  (訓練曲線)"
    
    # 顯示最後幾行日誌
    echo ""
    echo "最後 10 行訓練日誌:"
    tail -10 "$LOG_FILE"
else
    echo ""
    echo "============================================"
    echo "❌ 訓練失敗！"
    echo "============================================"
    echo "請查看日誌檔案: $LOG_FILE"
    echo ""
    echo "錯誤資訊的最後 20 行:"
    tail -20 "$LOG_FILE"
    exit 1
fi

echo ""
echo "實驗完成時間: $(date)"
echo "============================================"

#!/bin/bash

# ========================================
# 實驗 1: 驗證 Dropout 影響
# ========================================
# 假設: Dropout=0.3 導致資訊損失，減緩訓練速度
# 預期: Dropout=0 應能在 200 epochs 達到更高 accuracy

set -e

echo "🔬 實驗 1: 移除 Dropout (Dropout=0)"
echo "========================================"

# 激活環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 實驗編號
EXP_ID="exp1_no_dropout_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="../logs/${EXP_ID}.log"
OUTPUT_DIR="../results/${EXP_ID}"

mkdir -p ../logs "$OUTPUT_DIR"

# 自動選擇 GPU
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    awk '$2 > 8000 && $1 != 2 {print $1}' | head -1)

if [ -z "$AVAILABLE_GPUS" ]; then
    echo "❌ 沒有找到可用的 GPU"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS
echo "✅ 使用 GPU: $AVAILABLE_GPUS"

# ========================================
# 實驗參數設置
# ========================================
BATCH_SIZE=14
NUM_EPOCHS=200
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.01      # 保持與原版一致
D_MODEL=512
NHEAD=8
NUM_LAYERS=4
DROPOUT=0              # ⭐ 關鍵變數: 移除 Dropout

# 損失權重（與原版一致）
CE_WEIGHT=1.0
CONTENT_WEIGHT=0.0
EMBED_WEIGHT=0.0

# DataLoader 設置
CONTENT_RATIO=0.0
MIN_CONTENT_SAMPLES=2
MAX_SENTENCES=288
NUM_WORKERS=4
WARMUP_EPOCHS=10

echo ""
echo "📊 實驗配置:"
echo "  Dropout: $DROPOUT (vs 原版 0.3) ⭐"
echo "  Weight Decay: $WEIGHT_DECAY (保持不變)"
echo "  Learning Rate: $LEARNING_RATE (保持不變)"
echo "  Num Layers: $NUM_LAYERS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# 執行訓練
python -u train_token_denoising_hybrid.py \
    --input_dirs ../data/raw/box \
    --target_dir ../data/clean/box2 \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --d_model ${D_MODEL} \
    --nhead ${NHEAD} \
    --num_layers ${NUM_LAYERS} \
    --dropout ${DROPOUT} \
    --ce_weight ${CE_WEIGHT} \
    --content_weight ${CONTENT_WEIGHT} \
    --embed_weight ${EMBED_WEIGHT} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --content_ratio ${CONTENT_RATIO} \
    --min_content_samples ${MIN_CONTENT_SAMPLES} \
    --max_sentences_per_speaker ${MAX_SENTENCES} \
    --wavtokenizer_config ../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --wavtokenizer_checkpoint ../models/wavtokenizer_large_speech_320_24k.ckpt \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 實驗 1 完成！"
    echo ""
    echo "📊 驗證指標："
    echo "  - 檢查 epoch 200 的 Train Accuracy"
    echo "  - 與原版 (Dropout=0.3) 比較收斂速度"
    echo ""
    echo "📁 結果位置："
    echo "  - 日誌: $LOG_FILE"
    echo "  - 輸出: $OUTPUT_DIR"
    echo ""
    echo "🔍 快速查看結果："
    echo "  grep 'Epoch 200' $LOG_FILE"
else
    echo "❌ 實驗 1 失敗 (exit code: $EXIT_CODE)"
fi
echo "========================================"

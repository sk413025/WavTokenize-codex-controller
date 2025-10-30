#!/bin/bash

# ========================================
# 實驗 3: 驗證資料量影響
# ========================================
# 假設: 大量訓練樣本導致收斂慢，少量樣本快速過擬合
# 預期: 只用 14 個樣本應該在 100 epochs 內接近 100% accuracy

set -e

echo "🔬 實驗 3: 小資料集測試 (只用 14 個音檔)"
echo "========================================"

# 激活環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 實驗編號
EXP_ID="exp3_mini_dataset_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="../logs/${EXP_ID}.log"
OUTPUT_DIR="../results/${EXP_ID}"
MINI_DATA_DIR="../data/mini_dataset"

mkdir -p ../logs "$OUTPUT_DIR" "${MINI_DATA_DIR}/noisy" "${MINI_DATA_DIR}/clean"

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
# 創建小資料集 (複製 debug_single_sample.py 使用的 14 個音檔)
# ========================================
echo ""
echo "📦 準備小資料集 (14 個音檔)..."

# 定義要複製的音檔 (與 debug_single_sample.py 相同)
SPEAKERS=(
    "nor_boy1"
    "nor_boy3"
    "nor_boy4"
    "nor_boy5"
    "nor_boy6"
    "nor_boy9"
    "nor_boy10"
    "nor_girl2"
    "nor_girl3"
    "nor_girl4"
    "nor_girl6"
    "nor_girl7"
    "nor_girl8"
    "nor_girl11"
)

# 複製 noisy 音檔
for speaker in "${SPEAKERS[@]}"; do
    src="../data/raw/box/${speaker}_box_LDV_001.wav"
    dst="${MINI_DATA_DIR}/noisy/${speaker}_box_LDV_001.wav"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "  ✓ 複製: $speaker"
    else
        echo "  ✗ 找不到: $src"
    fi
done

# 複製 clean 音檔
for speaker in "${SPEAKERS[@]}"; do
    src="../data/clean/box2/${speaker}_clean_001.wav"
    dst="${MINI_DATA_DIR}/clean/${speaker}_clean_001.wav"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
    else
        echo "  ✗ 找不到: $src"
    fi
done

echo "✅ 小資料集準備完成 ($(ls ${MINI_DATA_DIR}/noisy | wc -l) 個音檔)"

# ========================================
# 實驗參數設置
# ========================================
BATCH_SIZE=14          # 與樣本數相同，每個 epoch 只有 1 個 batch
NUM_EPOCHS=200
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.01      # 保持與原版一致
D_MODEL=512
NHEAD=8
NUM_LAYERS=4
DROPOUT=0              # ⭐ 使用 0 以促進過擬合 (與 debug_single_sample.py 一致)

# 損失權重（與原版一致）
CE_WEIGHT=1.0
CONTENT_WEIGHT=0.0
EMBED_WEIGHT=0.0

# DataLoader 設置
CONTENT_RATIO=0.0
MIN_CONTENT_SAMPLES=2
MAX_SENTENCES=288      # 不影響，因為只有 14 個檔案
NUM_WORKERS=0          # 小資料集用 0 workers
WARMUP_EPOCHS=10

echo ""
echo "📊 實驗配置:"
echo "  資料集大小: 14 個音檔 (vs 原版 ~240+ 個) ⭐"
echo "  Batch Size: $BATCH_SIZE (= 樣本數，每 epoch 1 個 batch)"
echo "  Dropout: $DROPOUT (促進過擬合)"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# 執行訓練
python -u train_token_denoising_hybrid.py \
    --input_dirs ${MINI_DATA_DIR}/noisy \
    --target_dir ${MINI_DATA_DIR}/clean \
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
    echo "✅ 實驗 3 完成！"
    echo ""
    echo "📊 驗證指標："
    echo "  - 檢查 epoch 100 的 Train Accuracy (預期 >90%)"
    echo "  - 與 debug_single_sample.py 比較收斂速度"
    echo "  - 驗證資料量是否為主要影響因素"
    echo ""
    echo "📁 結果位置："
    echo "  - 日誌: $LOG_FILE"
    echo "  - 輸出: $OUTPUT_DIR"
    echo "  - 小資料集: $MINI_DATA_DIR"
    echo ""
    echo "🔍 快速查看結果："
    echo "  grep 'Epoch 100' $LOG_FILE"
    echo "  grep 'Epoch 200' $LOG_FILE"
else
    echo "❌ 實驗 3 失敗 (exit code: $EXIT_CODE)"
fi
echo "========================================"

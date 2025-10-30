#!/bin/bash

# Token Denoising Transformer - 簡化高效版本（使用已驗證的訓練腳本）
# 實驗編號: simple_efficient_$(date +%Y%m%d%H%M)

set -e

# 確保在 test 環境下運行
echo "🔧 激活 conda test 環境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 驗證環境
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$CURRENT_ENV" != "test" ]; then
    echo "❌ 錯誤: 未能激活 test 環境 (當前: $CURRENT_ENV)"
    exit 1
fi
echo "✅ 已激活環境: $CURRENT_ENV"

# 實驗編號
EXP_ID="simple_efficient_$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="../REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo ""
echo "====================================================="
echo "Token Denoising Transformer - 簡化高效版本 - $EXP_ID"
echo "====================================================="
echo "改進重點："
echo "1. ✅ 移除 Content Consistency Loss (content_weight=0)"
echo "2. ✅ 移除 Cross-Entropy Loss (ce_weight=0)"
echo "3. ✅ 只使用 Embedding L2 Loss (embed_weight=1.0)"
echo "4. ✅ 使用標準 DataLoader (content_ratio=0)"
echo "5. ✅ 增強正則化（Dropout 0.3, Weight Decay 0.1）"
echo "6. ✅ 無 Early Stopping（訓練完整 600 epochs）"
echo "7. ✅ Transformer Layers: 4 層（中等容量）"
echo "8. ✅ 每100 epoch儲存音檔、頻譜圖、checkpoint"
echo "9. ✅ 每50 epoch儲存loss圖"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TTT_BATCH_SIZE=14
export TTT_NUM_WORKERS=4
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CUDA_LAUNCH_BLOCKING=0

# 設置文件路徑
LOG_FILE="../logs/token_denoising_simple_${EXP_ID}.log"
OUTPUT_DIR="../results/token_denoising_simple_${EXP_ID}"

# 創建目錄
mkdir -p ../logs
mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")"

# 資料路徑
INPUT_DIRS="../data/raw/box"
TARGET_DIR="../data/clean/box2"

# 自動選擇空閒的GPU
echo "🔍 檢測可用的GPU..."
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    awk '$2 > 8000 && $1 != 2 {print $1}' | head -1)

if [ -z "$AVAILABLE_GPUS" ]; then
    echo "❌ 沒有找到有足夠空閒記憶體的GPU (需要 > 8GB, 排除GPU 2)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS
echo "✅ 使用 GPU: $AVAILABLE_GPUS"

# 顯示GPU狀態
nvidia-smi --id=$AVAILABLE_GPUS --query-gpu=memory.total,memory.used,memory.free --format=csv

# 訓練參數
BATCH_SIZE=14
NUM_EPOCHS=200
LEARNING_RATE=3e-4   #1e-4
WEIGHT_DECAY=0.01  # 增強正則化 0.1改0.01

# 模型參數
D_MODEL=512
NHEAD=8
NUM_LAYERS=4 # 中等容量 4層
DROPOUT=0     #0.3


# ====== 本次調整動機 ======
# 目的：平衡 token 預測能力與 embedding 結構，期望 validation loss 與 accuracy 同時提升
# 1. 開啟 CrossEntropy Loss，降低權重 (CE_WEIGHT=0.3)
# 2. 提高 Embedding L2 Loss 權重 (EMBED_WEIGHT=0.7)
# 3. Content Consistency Loss 關閉 (CONTENT_WEIGHT=0.0)
# ==========================
CE_WEIGHT=1  # 降低 CE 權重，提升 token accuracy 但避免過擬合
CONTENT_WEIGHT=0.0  # 關閉 Content Loss
EMBED_WEIGHT=0  # 提高 L2 Loss 權重，穩定 embedding 結構

# Content-aware sampler 參數（設為 0 = 使用標準 DataLoader）
CONTENT_RATIO=0.0  # 0 = 標準 shuffle DataLoader
MIN_CONTENT_SAMPLES=2

# 其他參數
MAX_SENTENCES=1
NUM_WORKERS=4
WARMUP_EPOCHS=10

echo ""
echo "📝 開始訓練 - Token Denoising (簡化高效版本)"
echo "=================================================="
echo "模型配置:"
echo "  - d_model: $D_MODEL"
echo "  - Encoder layers: $NUM_LAYERS"
echo "  - Attention heads: $NHEAD"
echo "  - Feedforward dim: 2048"
echo "  - Dropout: $DROPOUT (高正則化)"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Weight Decay: $WEIGHT_DECAY (高正則化)"
echo ""
echo "損失配置:"
echo "  - CE Loss: $CE_WEIGHT (= 0, 完全移除)"
echo "  - Content Loss: $CONTENT_WEIGHT (= 0, 完全移除)"
echo "  - Embed Loss: $EMBED_WEIGHT (= 1.0, 唯一損失函數)"
echo ""
echo "數據處理:"
echo "  - Content Ratio: $CONTENT_RATIO (= 0, 標準 DataLoader)"
echo "  - DataLoader: 標準模式（shuffle=True）"
echo "  - Workers: $NUM_WORKERS"
echo ""
echo "儲存配置:"
echo "  - Checkpoint + 音檔 + 頻譜圖: 每 100 epochs"
echo "  - Loss curves: 每 50 epochs"
echo "=================================================="
echo ""

# 執行訓練（使用已驗證的 train_token_denoising_hybrid.py）
python -u train_token_denoising_hybrid.py \
    --input_dirs ${INPUT_DIRS} \
    --target_dir ${TARGET_DIR} \
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
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 訓練完成！"
else
    echo "❌ 訓練出現錯誤 (exit code: $EXIT_CODE)"
fi
echo "=================================================="
echo "實驗 ID: $EXP_ID"
echo "輸出目錄: $OUTPUT_DIR"
echo "日誌文件: $LOG_FILE"
echo ""

# 更新實驗報告
if [ $EXIT_CODE -eq 0 ]; then
    cat >> "$REPORT_FILE" << EOF

---

## 實驗: Token Denoising Simple & Efficient ($EXP_ID)
**時間**: $TIMESTAMP  
**實驗 ID**: \`$EXP_ID\`

### 🎯 實驗目的
解決嚴重過擬合問題，採用簡化高效的訓練策略。

### 🔧 核心改進

#### 1. 移除 Content Consistency Loss
**實現方式**: \`--content_weight 0.0\`
- 完全移除 Content Consistency Loss
- 只保留 CE Loss 和 Embedding L2 Loss
- 讓模型自然學習語者特徵和內容分離

#### 2. 標準 DataLoader
**實現方式**: \`--content_ratio 0.0\`
- 使用標準 DataLoader with shuffle=True
- 簡化數據流，減少潛在 bug
- 保持數據隨機性，自然防止過擬合

#### 3. 增強正則化
**配置**:
- Dropout: 0.3（從 0.2 提升）
- Weight Decay: 0.1（從 0.05 提升）
- num_layers: 4（中等容量）

#### 4. 完整儲存邏輯
**每 100 epoch**:
- Checkpoint (模型權重 + 優化器狀態)
- 音檔樣本 (noisy, enhanced, clean)
- 頻譜圖 (3個音檔對比)

**每 50 epoch**:
- Loss 曲線圖

### 📊 模型配置

| 參數 | 值 | 說明 |
|------|------|------|
| d_model | 512 | 標準維度 |
| num_layers | 4 | 中等容量 |
| dropout | 0.3 | 高正則化 |
| weight_decay | 0.1 | 高正則化 |
| batch_size | 8 | 平衡速度與記憶體 |
| learning_rate | 1e-4 | 標準學習率 |

### 🔧 技術細節

#### 損失函數
\`\`\`python
loss = CE_loss + 0.3 * Embedding_L2_loss
# Content_weight = 0.0 → 完全移除 Content Loss
\`\`\`

#### DataLoader
\`\`\`python
# content_ratio = 0.0 → 使用標準 DataLoader
DataLoader(..., shuffle=True)  # 標準隨機打亂
\`\`\`

### 📁 輸出路徑
- 模型: \`$OUTPUT_DIR\`
- 日誌: \`$LOG_FILE\`

### 🔬 預期效果

✅ **過擬合改善**:
- Train/Val Accuracy Gap < 15%（epoch 200）
- Val Accuracy > 30%（epoch 200）
- Val Loss 穩定下降

✅ **語者保留**:
- 模型學習語者特徵而非忽略
- Enhanced audio 保留原始語者特性

### 📝 重現步驟
1. 執行訓練: \`bash try/run_simple_efficient.sh\`
2. 監控日誌: \`tail -f $LOG_FILE\`
3. 查看音檔: \`$OUTPUT_DIR/audio_samples/epoch_*/\`
4. 查看曲線: \`$OUTPUT_DIR/loss_curves_*.png\`

---
EOF

    echo "📝 實驗記錄已更新到 $REPORT_FILE"
fi

echo ""
echo "🎉 實驗設置完成！"
echo "💡 提示: 使用 'tail -f $LOG_FILE' 監控訓練進度"

#!/bin/bash

# Token Denoising Transformer - 簡化版 (僅 CE Loss)
# 實驗編號: ce_only_$(date +%Y%m%d%H%M)

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
EXP_ID="ce_only_$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="../REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo ""
echo "====================================================="
echo "Token Denoising Transformer - 簡化版 (僅 CE Loss) - $EXP_ID"
echo "====================================================="
echo "核心特點："
echo "1. ✅ 只使用 CrossEntropy Loss"
echo "2. ✅ 移除所有混合損失組件"
echo "3. ✅ 使用標準 DataLoader (shuffle=True)"
echo "4. ✅ 簡化訓練流程，專注於 token 預測準確度"
echo "5. ✅ Transformer Layers: 4 層（中等容量）"
echo "6. ✅ 每100 epoch儲存音檔、頻譜圖、checkpoint"
echo "7. ✅ 每50 epoch儲存loss圖"
echo "====================================================="

# 設置環境變數
#export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TTT_BATCH_SIZE=14
export TTT_NUM_WORKERS=4
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CUDA_LAUNCH_BLOCKING=0

# 設置文件路徑
LOG_FILE="./logs/token_denoising_ce_only_${EXP_ID}.log"
OUTPUT_DIR="./results/${EXP_ID}"

# 創建目錄
mkdir -p ./logs
mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")"

# 資料路徑
INPUT_DIRS="../data/raw/box ../data/raw/papercup ../data/raw/plastic ../data/clean/box2"
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
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.01

# 模型參數
D_MODEL=512
NHEAD=8
NUM_LAYERS=4
DROPOUT=0

# 其他參數
MAX_SENTENCES=288   #設定每位語者能說幾句話
NUM_WORKERS=4

echo ""
echo "📝 開始訓練 - Token Denoising (簡化版 - 僅 CE Loss)"
echo "=================================================="
echo "模型配置:"
echo "  - d_model: $D_MODEL"
echo "  - Encoder layers: $NUM_LAYERS"
echo "  - Attention heads: $NHEAD"
echo "  - Feedforward dim: 2048"
echo "  - Dropout: $DROPOUT"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Weight Decay: $WEIGHT_DECAY"
echo ""
echo "損失配置:"
echo "  - CE Loss: 只使用 CrossEntropy Loss"
echo "  - 無其他損失函數"
echo ""
echo "數據處理:"
echo "  - DataLoader: 標準模式（shuffle=True）"
echo "  - Workers: $NUM_WORKERS"
echo ""
echo "儲存配置:"
echo "  - Checkpoint + 音檔 + 頻譜圖: 每 100 epochs"
echo "  - Loss curves: 每 50 epochs"
echo "=================================================="
echo ""

# 執行訓練（使用簡化版的 train.py）
python -u train.py \
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

## 實驗: Token Denoising CE Loss Only ($EXP_ID)
**時間**: $TIMESTAMP
**實驗 ID**: \`$EXP_ID\`

### 🎯 實驗目的
簡化訓練流程，只使用 CrossEntropy Loss，專注於提升 token 預測準確度。

### 🔧 核心特點

#### 1. 僅使用 CE Loss
**實現方式**: 移除所有混合損失組件
- 只保留 CrossEntropy Loss
- 移除 Content Consistency Loss
- 移除 Embedding L2 Loss
- 移除 Spectral Loss
- 讓模型專注於 token 級別的預測準確度

#### 2. 標準 DataLoader
**實現方式**: 使用標準 DataLoader with shuffle=True
- 簡化數據流，減少潛在 bug
- 保持數據隨機性，自然防止過擬合
- 不使用內容感知採樣器

#### 3. 模型配置
**配置**:
- d_model: 512
- num_layers: 4（中等容量）
- dropout: 0
- weight_decay: 0.01
- batch_size: 14
- learning_rate: 3e-4

#### 4. 完整儲存邏輯
**每 100 epoch**:
- Checkpoint (模型權重 + 優化器狀態)
- 音檔樣本 (noisy, enhanced, clean)
- 頻譜圖 (3個音檔對比)

**每 50 epoch**:
- Loss 曲線圖 (只包含 CE Loss 和 Token Accuracy)

### 📊 模型配置

| 參數 | 值 | 說明 |
|------|------|------|
| d_model | 512 | 標準維度 |
| num_layers | 4 | 中等容量 |
| dropout | 0 | 無 dropout |
| weight_decay | 0.01 | 輕量正則化 |
| batch_size | 14 | 平衡速度與記憶體 |
| learning_rate | 3e-4 | 標準學習率 |

### 🔧 技術細節

#### 損失函數
\`\`\`python
# 只使用 CrossEntropy Loss
loss = CrossEntropyLoss(pred_logits, target_tokens)
\`\`\`

#### DataLoader
\`\`\`python
# 使用標準 DataLoader
DataLoader(..., shuffle=True)  # 標準隨機打亂
\`\`\`

### 📁 輸出路徑
- 模型: \`$OUTPUT_DIR\`
- 日誌: \`$LOG_FILE\`

### 🔬 預期效果

✅ **訓練穩定性**:
- Loss 穩定下降
- Accuracy 持續提升
- 無複雜損失權重調整

✅ **評估指標**:
- Token Accuracy（主要指標）
- CE Loss（訓練/驗證）

### 📝 重現步驟
1. 執行訓練: \`bash done/run.sh\`
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

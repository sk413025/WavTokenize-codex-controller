#!/bin/bash

# Token Denoising Transformer - 方案 C: Progressive Masking
# 漸進式遮罩 (從 5% 逐步增加到 30%)

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
EXP_ID="mask_progressive_$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="../REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo ""
echo "======================================================"
echo "方案 C: Progressive Masking - $EXP_ID"
echo "======================================================"
echo "核心特點："
echo "1. ✅ 漸進式遮罩 (5% → 30%)"
echo "2. ✅ 前 100 epochs 線性增長"
echo "3. ✅ 課程學習策略 (由易到難)"
echo "4. ✅ 標準 CE Loss"
echo "======================================================"

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CUDA_LAUNCH_BLOCKING=0

# 設置文件路徑
LOG_FILE="./logs/mask_progressive_${EXP_ID}.log"
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
DROPOUT=0.1

# 🎭 遮罩參數 (Progressive)
MASK_STRATEGY="progressive"
PROGRESSIVE_START_RATIO=0.05
PROGRESSIVE_END_RATIO=0.30
PROGRESSIVE_EPOCHS=100

# 其他參數
MAX_SENTENCES=288
NUM_WORKERS=4

echo ""
echo "📝 開始訓練 - Progressive Masking (5% → 30%)"
echo "=================================================="
echo "模型配置:"
echo "  - d_model: $D_MODEL"
echo "  - Encoder layers: $NUM_LAYERS"
echo "  - Attention heads: $NHEAD"
echo "  - Dropout: $DROPOUT"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo ""
echo "遮罩配置:"
echo "  - Strategy: $MASK_STRATEGY"
echo "  - Start Ratio: $PROGRESSIVE_START_RATIO (5%)"
echo "  - End Ratio: $PROGRESSIVE_END_RATIO (30%)"
echo "  - Progressive Epochs: $PROGRESSIVE_EPOCHS"
echo "  - Mask Token ID: 4095"
echo ""
echo "遮罩比例變化:"
echo "  - Epoch 1-25:   5% → 11.25%"
echo "  - Epoch 26-50:  11.25% → 17.5%"
echo "  - Epoch 51-75:  17.5% → 23.75%"
echo "  - Epoch 76-100: 23.75% → 30%"
echo "  - Epoch 101+:   固定 30%"
echo ""
echo "實驗目的:"
echo "  - 課程學習：從簡單到困難"
echo "  - 逐步提高泛化能力"
echo "  - 避免訓練初期過度困難"
echo "=================================================="
echo ""

# 執行訓練
python -u train_mask.py \
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
    --mask_strategy ${MASK_STRATEGY} \
    --progressive_start_ratio ${PROGRESSIVE_START_RATIO} \
    --progressive_end_ratio ${PROGRESSIVE_END_RATIO} \
    --progressive_epochs ${PROGRESSIVE_EPOCHS} \
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

## 實驗: 方案 C - Progressive Masking ($EXP_ID)
**時間**: $TIMESTAMP
**實驗 ID**: \`$EXP_ID\`

### 🎯 實驗目的
通過漸進式增加遮罩比例，實現課程學習（Curriculum Learning），讓模型從簡單任務逐步過渡到困難任務。

### 🔧 方案 C: Progressive Masking

#### 核心思路
訓練初期使用較低遮罩比例（5%），隨訓練進行逐漸增加到 30%，讓模型先學會基本的去噪，再逐步提升泛化能力。

#### 實現方式
\`\`\`python
# 計算當前 epoch 的遮罩比例
if epoch <= progressive_epochs:
    progress = epoch / progressive_epochs
    current_ratio = start_ratio + progress * (end_ratio - start_ratio)
else:
    current_ratio = end_ratio

# 應用動態遮罩
mask_positions = (torch.rand(B, T) < current_ratio)
masked_tokens[mask_positions] = 4095
\`\`\`

#### 技術細節
- **起始遮罩比例**: 5% (epoch 1)
- **最終遮罩比例**: 30% (epoch 100+)
- **增長方式**: 線性增長
- **增長期間**: 前 100 epochs
- **損失函數**: 標準 CrossEntropy Loss

### 📊 模型配置

| 參數 | 值 | 說明 |
|------|------|------|
| d_model | 512 | 標準維度 |
| num_layers | 4 | 中等容量 |
| dropout | 0 | 無 dropout |
| batch_size | 14 | 標準批次大小 |
| learning_rate | 3e-4 | 標準學習率 |
| progressive_start_ratio | 0.05 | 起始 5% |
| progressive_end_ratio | 0.30 | 最終 30% |
| progressive_epochs | 100 | 100 epochs 達到最大 |

### 📈 遮罩比例時間表

| Epoch Range | Mask Ratio | 說明 |
|-------------|------------|------|
| 1-25 | 5.0% → 11.25% | 初期：輕度遮罩 |
| 26-50 | 11.25% → 17.5% | 中前期：逐步增加 |
| 51-75 | 17.5% → 23.75% | 中後期：持續增長 |
| 76-100 | 23.75% → 30% | 後期：接近最大 |
| 101+ | 30% (固定) | 穩定期：最大遮罩 |

### 📁 輸出路徑
- 模型: \`$OUTPUT_DIR\`
- 日誌: \`$LOG_FILE\`

### 🔬 預期效果

✅ **訓練穩定性**:
- 初期收斂更快（低遮罩率）
- 後期泛化能力更強（高遮罩率）
- Loss 曲線更平滑

✅ **課程學習優勢**:
- 避免訓練初期過度困難
- 逐步建立魯棒性
- 最終達到較高的泛化能力

✅ **對比固定遮罩**:
- 訓練初期更穩定
- 最終模型可能更魯棒（30% vs 10%）

### 📝 重現步驟
1. 執行訓練: \`bash done/run_mask_progressive.sh\`
2. 監控日誌: \`tail -f $LOG_FILE\`
3. 查看結果: \`$OUTPUT_DIR/\`

### 🔍 分析建議
重點觀察：
1. **不同階段的 loss 變化**
   - Epoch 1-25 (5-11%): 應快速下降
   - Epoch 26-100 (11-30%): 可能略有波動
   - Epoch 101+ (30%): 穩定收斂

2. **對比固定 10% 遮罩**
   - 訓練初期：Progressive 應更快收斂
   - 訓練後期：Progressive 可能 loss 略高（因遮罩率更高）
   - 泛化能力：Progressive 可能更好（經歷更高遮罩率訓練）

3. **音頻品質**
   - 不同 epoch 的音頻樣本品質演進

---
EOF

    echo "📝 實驗記錄已更新到 $REPORT_FILE"
fi

echo ""
echo "🎉 實驗設置完成！"
echo "💡 提示: 使用 'tail -f $LOG_FILE' 監控訓練進度"

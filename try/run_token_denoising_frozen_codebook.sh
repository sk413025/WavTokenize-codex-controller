#!/bin/bash

# Token Denoising Transformer with Frozen Codebook
# 完全凍結 WavTokenizer Codebook，類比機器翻譯的 Frozen Embedding
# 實驗編號: frozen_codebook_$(date +%Y%m%d%H%M)

set -e

# 實驗編號
EXP_ID="frozen_codebook_$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="../REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "====================================================="
echo "Token Denoising Transformer - Frozen Codebook - $EXP_ID"
echo "====================================================="
echo "核心理念："
echo "1. ✅ 完全凍結 WavTokenizer Codebook (不訓練 embedding)"
echo "2. ✅ 只訓練 Transformer Encoder + Output Projection"
echo "3. ✅ Token-to-Token 映射 (類比機器翻譯)"
echo "4. ✅ 更輕量、更快收斂"
echo ""
echo "與現有模型的差異："
echo "  現有模型: 重新訓練 Codebook Embedding"
echo "  本模型:   直接查表 WavTokenizer Codebook (凍結)"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TTT_BATCH_SIZE=8  # Frozen Codebook 更節省記憶體
export TTT_NUM_WORKERS=4
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CONTENT_BATCHING=true
export CUDA_LAUNCH_BLOCKING=1

# 設置文件路徑
LOG_FILE="../logs/token_denoising_frozen_codebook_${EXP_ID}.log"
OUTPUT_DIR="../results/token_denoising_frozen_codebook_${EXP_ID}"

# 創建目錄
mkdir -p ../logs
mkdir -p "$OUTPUT_DIR"

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

echo ""
echo "📝 開始訓練 - Token Denoising with Frozen Codebook"
echo "=================================================="
echo "模型配置:"
echo "  - Codebook: 完全凍結 (WavTokenizer 預訓練)"
echo "  - d_model: 512 (與 Codebook 維度一致)"
echo "  - Encoder layers: 6"
echo "  - Attention heads: 8"
echo "  - Feedforward dim: 2048"
echo "  - Batch size: 8"
echo "  - Learning Rate: 1e-4"
echo "  - Loss: Cross-Entropy (Token-level)"
echo "=================================================="
echo ""

# 執行訓練
python -u train_token_denoising.py \
    --config ../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --model_path ../models/wavtokenizer_large_speech_320_24k.ckpt \
    --d_model 512 \
    --nhead 8 \
    --num_layers 6 \
    --dim_feedforward 2048 \
    --dropout 0.1 \
    --batch_size $TTT_BATCH_SIZE \
    --num_epochs 600 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 100 \
    --val_speakers girl9 girl10 boy7 boy8 \
    --train_speakers boy1 boy3 boy4 boy5 boy6 boy9 boy10 girl2 girl3 girl4 girl6 girl7 girl8 girl11 \
    2>&1 | tee -a $LOG_FILE

echo ""
echo "=================================================="
echo "✅ 訓練完成！"
echo "=================================================="
echo "實驗 ID: $EXP_ID"
echo "輸出目錄: $OUTPUT_DIR"
echo "日誌文件: $LOG_FILE"
echo ""

# 更新實驗報告
cat >> "$REPORT_FILE" << EOF

---

## 實驗: Token Denoising with Frozen Codebook ($EXP_ID)
**時間**: $TIMESTAMP  
**實驗 ID**: \`$EXP_ID\`

### 🎯 實驗目的
測試完全凍結 WavTokenizer Codebook 的降噪效果，類比機器翻譯的 Frozen Embedding 策略。

### 🔬 核心假設
1. **WavTokenizer 的 Codebook 已經學到最佳的音訊表示**
   - 不需要重新訓練 embedding
   - 直接查表即可獲得高質量的聲學特徵

2. **降噪 = Token-to-Token 的序列映射**
   - 輸入: Noisy Token IDs [0, 4095]
   - 輸出: Clean Token IDs [0, 4095]
   - 類比: 英文→中文翻譯

### 📊 模型架構對比

#### 現有模型 (wavtokenizer_transformer_denoising.py)
\`\`\`
Audio → Encoder → Tokens → [可訓練 Codebook Embedding] 
      → Transformer Encoder-Decoder → Decoder → Audio
\`\`\`
- 可訓練參數: ~5-6M (含 Codebook Embedding)

#### 本模型 (token_denoising_transformer.py - Frozen Codebook)
\`\`\`
Audio → Encoder → Tokens → [凍結 Codebook Lookup] 
      → Transformer Encoder → Output Projection → Tokens → Decoder → Audio
\`\`\`
- 可訓練參數: ~3-4M (不含 Codebook)

### 🔧 技術細節

| 組件 | 現有模型 | Frozen Codebook 模型 |
|------|----------|---------------------|
| Codebook Embedding | ✅ 可訓練 | ❌ 完全凍結 |
| Transformer 架構 | Encoder-Decoder | Encoder Only |
| Embedding 查表 | 需要學習映射 | 直接 Codebook 查表 |
| 參數量 | ~5-6M | ~3-4M |
| 記憶體佔用 | 較高 | 較低 |

### 🎨 設計靈感
類比 **機器翻譯的 Frozen Pretrained Embedding**：
- 英文詞嵌入 (frozen) → Transformer → 中文詞 IDs
- Noisy Token IDs → Frozen Codebook → Transformer → Clean Token IDs

### 📁 輸出路徑
- 模型: \`$OUTPUT_DIR\`
- 日誌: \`$LOG_FILE\`

### 🔬 預期效果
✅ **優勢**:
1. 更快收斂 (參數更少)
2. 更穩定訓練 (embedding 不變)
3. 更好的泛化 (利用預訓練知識)

⚠ **風險**:
1. Codebook 可能不完美適配降噪任務
2. 無法微調 embedding 以適應特定噪音類型

### 📊 評估指標
- Token 準確率 (與 Ground Truth 比較)
- Token 變化率 (降噪前後差異)
- 音訊質量 (PESQ, STOI)
- 頻譜相似度 (MSE, Correlation)

---
EOF

echo "📝 實驗記錄已更新到 $REPORT_FILE"

# 提交 Git Commit
echo ""
echo "📝 提交 Git Commit..."
git add -A
git commit -m "實驗 ${EXP_ID}: Token Denoising with Frozen Codebook

## 實驗背景
測試完全凍結 WavTokenizer Codebook 的降噪策略，類比機器翻譯的 Frozen Embedding。

## 實驗動機
現有模型重新訓練 Codebook Embedding，可能破壞 WavTokenizer 學到的聲學知識。
本實驗假設：預訓練的 Codebook 已經是最佳表示，不需要重新學習。

## 實驗目的
1. 驗證 Frozen Codebook 是否足以支持降噪任務
2. 比較與可訓練 Embedding 的效果差異
3. 測試更輕量模型的收斂速度和穩定性

## 預期結果
- Token 準確率: > 60% (epoch 200)
- 收斂速度: 比現有模型更快 (參數更少)
- 音訊質量: 與現有模型相當或更好

## 實際執行結果
(待訓練完成後更新)

## 解讀實驗結果
(待分析)

## 實驗反思
(待總結)

## 重現實驗步驟
1. 執行: bash run_token_denoising_frozen_codebook.sh
2. 監控: tail -f $LOG_FILE
3. 結果: 檢查 $OUTPUT_DIR
" || echo "⚠ Git commit 失敗（可能沒有變更）"

echo ""
echo "🎉 實驗設置完成！"
echo "💡 提示: 使用 'tail -f $LOG_FILE' 監控訓練進度"

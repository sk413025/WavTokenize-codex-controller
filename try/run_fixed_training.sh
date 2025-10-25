#!/bin/bash

# Token Denoising Transformer 修復後訓練腳本 - 防過擬合版本
# 實驗編號: fixed_anti_overfitting_$(date +%Y%m%d%H%M)

set -e

# 實驗編號
EXP_ID="fixed_anti_overfitting_$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="../REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "====================================================="
echo "Token Denoising Transformer - Fixed & Anti-Overfitting - $EXP_ID"
echo "====================================================="
echo "修復內容："
echo "1. ✅ 音頻儲存維度問題 (codes_to_features 4D -> 3D)"
echo "2. ✅ 新增 loss 曲線繪製 (每 50 epochs)"
echo "3. ✅ 應用防過擬合措施"
echo "   - dropout: 0.1 → 0.2"
echo "   - weight_decay: 0.01 → 0.05"
echo "   - num_layers: 6 → 4"
echo "   - 使用 ReduceLROnPlateau 學習率調度器"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TTT_BATCH_SIZE=8
export TTT_NUM_WORKERS=4
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CONTENT_BATCHING=true
export CUDA_LAUNCH_BLOCKING=1

# 設置文件路徑
LOG_FILE="../logs/token_denoising_fixed_${EXP_ID}.log"
OUTPUT_DIR="../results/token_denoising_fixed_${EXP_ID}"

# 創建目錄
mkdir -p ../logs
mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")"

# 資料路徑
INPUT_DIRS="../data/raw/box"  # 噪音音訊
TARGET_DIR="../data/clean/box2"  # 乾淨音訊

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
BATCH_SIZE=8
NUM_EPOCHS=600
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.05  # 提高以防止過擬合

# 模型參數（防過擬合配置）
D_MODEL=512
NHEAD=8
NUM_LAYERS=4  # 從 6 降至 4
DROPOUT=0.2  # 從 0.1 提高至 0.2

# 混合損失權重
CE_WEIGHT=1.0
CONTENT_WEIGHT=0.5
EMBED_WEIGHT=0.3
WARMUP_EPOCHS=10

# ContentAwareBatchSampler 參數
CONTENT_RATIO=0.5
MIN_CONTENT_SAMPLES=3

# 資料參數
MAX_SENTENCES=288  # 每個說話者最多使用的句子數

echo ""
echo "📝 開始訓練 - Token Denoising with Fixed Bugs & Anti-Overfitting"
echo "=================================================="
echo "模型配置:"
echo "  - d_model: $D_MODEL"
echo "  - Encoder layers: $NUM_LAYERS (降低)"
echo "  - Attention heads: $NHEAD"
echo "  - Feedforward dim: 2048"
echo "  - Dropout: $DROPOUT (提高)"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Weight Decay: $WEIGHT_DECAY (提高)"
echo "  - Scheduler: ReduceLROnPlateau"
echo ""
echo "損失配置:"
echo "  - CE Loss: $CE_WEIGHT"
echo "  - Content Loss: $CONTENT_WEIGHT"
echo "  - Embed Loss: $EMBED_WEIGHT"
echo "  - Warmup: $WARMUP_EPOCHS epochs"
echo ""
echo "儲存配置:"
echo "  - Checkpoint: 每 10 epochs"
echo "  - Audio samples: 每 100 epochs"
echo "  - Loss curves: 每 50 epochs"
echo "=================================================="
echo ""

# 執行訓練
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

## 實驗: Token Denoising Fixed & Anti-Overfitting ($EXP_ID)
**時間**: $TIMESTAMP  
**實驗 ID**: \`$EXP_ID\`

### 🎯 實驗目的
修復所有已知 bug 並應用防過擬合措施，重新訓練 Token Denoising Transformer。

### 🐛 修復內容
1. **音頻儲存維度問題**
   - 問題：codes_to_features() 返回 4D tensor [1, T, 1, D]，導致 decode() 失敗
   - 修復：自動檢測並轉換為 3D [1, T, D]
   - 影響：每 100 epochs 的音頻樣本現在可以正常儲存

2. **缺少 loss 曲線繪製**
   - 問題：沒有視覺化訓練過程
   - 修復：新增 plot_loss_curves() 函數
   - 影響：每 50 epochs 自動生成 4 個子圖的訓練曲線

3. **嚴重過擬合問題**
   - 問題：Train acc 79%, Val acc 22%, Test acc 0.42%
   - 修復：
     * dropout: 0.1 → 0.2
     * weight_decay: 0.01 → 0.05
     * num_layers: 6 → 4
     * scheduler: CosineAnnealing → ReduceLROnPlateau

### 📊 模型配置

| 參數 | 舊值 | 新值 | 原因 |
|------|------|------|------|
| dropout | 0.1 | 0.2 | 增強正則化 |
| weight_decay | 0.01 | 0.05 | 減少過擬合 |
| num_layers | 6 | 4 | 降低模型容量 |
| scheduler | CosineAnnealing | ReduceLROnPlateau | 根據驗證損失動態調整 |

### 🔧 技術細節
- **Batch Size**: 8
- **Learning Rate**: 1e-4
- **Loss**: CE (1.0) + Content (0.5) + Embed (0.3)
- **Warmup**: 10 epochs
- **Content Batching**: ratio=0.5, min_samples=3

### 📁 輸出路徑
- 模型: \`$OUTPUT_DIR\`
- 日誌: \`$LOG_FILE\`

### 🔬 預期效果
✅ **目標**:
1. 音頻樣本正常儲存（每 100 epochs）
2. Loss 曲線正常生成（每 50 epochs）
3. 驗證準確率 > 50%（epoch 200）
4. Train/Val accuracy 差距 < 20%（防止過擬合）

### 📊 評估指標
- Token 準確率（Train / Val / Test）
- 各損失分量（CE / Content / Embed）
- 學習率變化曲線
- 音訊質量（頻譜圖對比）

---
EOF

echo "📝 實驗記錄已更新到 $REPORT_FILE"

# 提交 Git Commit
echo ""
echo "📝 提交 Git Commit..."
git add -A
git commit -m "實驗 ${EXP_ID}: Token Denoising Fixed & Anti-Overfitting

## 實驗背景
前次訓練（epoch 420）出現嚴重過擬合：
- Train accuracy: 79.55%
- Val accuracy: 22.68%
- Test accuracy: 0.42%

同時發現 3 個關鍵 bug：
1. 音頻儲存維度錯誤（每 100 epochs 失敗）
2. 缺少 loss 曲線繪製（無法監控）
3. 模型過擬合（泛化能力極差）

## 實驗動機
需要完全重新訓練，因為：
1. 模型已經嚴重過擬合，無法泛化
2. 音頻樣本儲存失敗，無法評估實際效果
3. 缺少訓練曲線，無法診斷問題

## 實驗目的
1. 修復所有已知 bug，確保訓練過程穩定
2. 應用防過擬合措施，提升模型泛化能力
3. 重新訓練並持續監控，確保不再過擬合

## 預期結果
- 音頻樣本正常儲存（每 100 epochs）
- Loss 曲線正常生成（每 50 epochs）
- Epoch 200: Val accuracy > 50%, Train/Val gap < 20%
- Epoch 600: Val accuracy > 70%, 穩定收斂

## 實際執行結果
(待訓練完成後更新)

## 解讀實驗結果
(待分析)

## 實驗反思
(待總結)

## 重現實驗步驟
1. 修復代碼: train_token_denoising_hybrid.py
   - 音頻儲存維度處理
   - 新增 plot_loss_curves() 函數
   - 調整預設參數
2. 執行: bash run_fixed_training.sh
3. 監控: tail -f $LOG_FILE
4. 結果: 檢查 $OUTPUT_DIR
" || echo "⚠ Git commit 失敗（可能沒有變更）"

echo ""
echo "🎉 實驗設置完成！"
echo "💡 提示: 使用 'tail -f $LOG_FILE' 監控訓練進度"

#!/bin/bash

# WavTokenizer-Transformer 大模型 + Token Loss
# 結合模型容量提升和聲學損失函數
# 修復版：解決 Learning Rate 過小問題
set -e

# 實驗編號
EXP_ID="large_tokenloss_FIXED_LR_$(date +%Y%m%d%H%M)"
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "====================================================="
echo "WavTokenizer-Transformer 大模型 + Token Loss (LR 修復版) - $EXP_ID"
echo "====================================================="
echo "改進策略："
echo "1. ✅ 模型容量提升："
echo "   - d_model: 128 → 256"
echo "   - layers: 2 → 4"
echo "   - dim_feedforward: 256 → 1024"
echo "   - nhead: 2 → 8"
echo "2. ✅ 使用混合 Token Loss (CE + L2_Embed + Coherence + Manifold)"
echo "   - CE Loss (15.0): 主要監督信號，確保預測準確 [修復：1.0→10.0→15.0]"
echo "   - L2 Embed (1.5): 聲學相似性，保持語者特徵 [修復：0.5→1.5]"
echo "   - Coherence (0.2): 時間平滑，解決頻譜破碎"
echo "   - Manifold (0.1): 正則化，防止偏離太遠"
echo "3. 🔧 Learning Rate 修復："
echo "   - LR: 5e-5 → 1e-4 (提升 2 倍)"
echo "   - 禁用 OneCycleLR scheduler (導致 LR 降到 2e-6)"
echo "   - 使用固定 LR，避免訓練後期學習停滯"
echo "   - 修復原因：Token 準確率 0%，CE Loss 停在 8.59 (接近隨機)"
echo "4. ✅ 預期效果："
echo "   - 前 50 epochs: Token Accuracy 0% → 10-20%"
echo "   - 100 epochs: Token Accuracy > 30%, CE Loss < 4.0"
echo "   - 200 epochs: Token Accuracy > 60%, CE Loss < 2.0"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TTT_BATCH_SIZE=4  # 大模型需要減少 batch size
export TTT_NUM_WORKERS=2
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CONTENT_BATCHING=true
export CUDA_LAUNCH_BLOCKING=1
export EFFECTIVE_BATCH_SIZE=8  # 使用梯度累積達到有效 batch size 8

# 設置文件路徑
LOG_FILE="logs/transformer_large_tokenloss_${EXP_ID}.log"
OUTPUT_DIR="results/transformer_large_tokenloss_${EXP_ID}"

# 創建目錄
mkdir -p logs
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
echo "📝 開始訓練 - 大模型 + Token Loss (LR 修復版)"
echo "=================================================="
echo "模型配置:"
echo "  - d_model: 256"
echo "  - Encoder/Decoder layers: 4"
echo "  - Feedforward dim: 1024"
echo "  - Attention heads: 8"
echo "  - Batch size: 4 (gradient accumulation: 2 → effective 8)"
echo "  - Learning Rate: 1e-4 (固定，無 scheduler)"
echo "  - Loss: Mixed Token Loss (CE=15.0 + L2_Embed=1.5 + Coherence=0.2 + Manifold=0.1)"
echo "=================================================="
echo ""

# 執行訓練（添加 -u 參數實現 unbuffered 輸出）
python -u wavtokenizer_transformer_denoising.py \
    --d_model 256 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --dim_feedforward 1024 \
    --max_length 400 \
    --batch_size $TTT_BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --num_epochs 1000 \
    --learning_rate 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 100 \
    --val_speakers girl9 girl10 boy7 boy8 \
    --train_speakers boy1 boy3 boy4 boy5 boy6 boy9 boy10 girl2 girl3 girl4 girl6 girl7 girl8 girl11 \
    --use_token_loss \
    --ce_weight 15.0 \
    --l2_embed_weight 1.5 \
    --coherence_weight 0.2 \
    --manifold_weight 0.1 \
    --disable_scheduler \
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

## 實驗: 大模型 + Token Loss ($EXP_ID)
**時間**: $TIMESTAMP  
**實驗 ID**: \`$EXP_ID\`

### 🎯 實驗目的
解決小模型頻譜重建不連續的問題，採用雙重策略：
1. 大幅提升模型容量
2. 使用 Token Loss 優化聲學結構

### 📊 模型配置
| 參數 | 舊值 | 新值 | 提升倍數 |
|------|------|------|----------|
| d_model | 128 | 256 | 2x |
| Encoder Layers | 2 | 4 | 2x |
| Decoder Layers | 2 | 4 | 2x |
| Feedforward Dim | 256 | 1024 | 4x |
| Attention Heads | 2 | 8 | 4x |
| Batch Size | 8 | 4 | 0.5x (記憶體限制) |
| Gradient Accum | 2 | 2 | - |

### 🔧 技術改進
1. **模型容量提升**
   - 可訓練參數: 1.26M → ~5-6M (預估)
   - 更強的長時依賴建模能力
   - 更好的 token 間關係學習

2. **Token Loss 引入**
   - L2 Loss: 直接優化 embedding 距離
   - Consistency Loss: 保證相鄰 token 的平滑過渡
   - 避免離散 token 預測的量化誤差

3. **訓練策略調整**
   - 學習率降低: 1e-4 → 5e-5 (大模型需要更小步長)
   - Batch size 減半: 記憶體考量
   - 保持有效 batch size 8 (梯度累積)

### 📁 輸出路徑
- 模型: \`$OUTPUT_DIR\`
- 日誌: \`$LOG_FILE\`

### 🔬 預期效果
- ✅ 更平滑的頻譜重建
- ✅ 更快的收斂速度
- ✅ 更好的聽覺質量
- ✅ 更少的頻譜破碎現象

---
EOF

echo "📝 實驗記錄已更新到 $REPORT_FILE"

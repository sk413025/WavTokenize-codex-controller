#!/bin/bash

# WavTokenizer-Transformer Denoising with Codebook Embedding
# 架構改進：使用預訓練 Codebook 而非隨機初始化 Embedding
set -e

# 實驗編號
EXP_ID="codebook_emb_$(date +%Y%m%d%H%M)"
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "====================================================="
echo "WavTokenizer-Transformer Codebook Embedding 實驗 - $EXP_ID"
echo "====================================================="
echo "架構改進："
echo "1. ✅ 使用 WavTokenizer 預訓練 Codebook 作為 Token Embedding"
echo "2. ✅ Codebook tokens (0-4095): 使用預訓練表示"
echo "3. ✅ Special tokens (pad/sos/eos): 使用可學習 embeddings"
echo "4. ✅ 保留預訓練模型的語義結構"
echo "5. ✅ 預期：更快收斂、更好性能、更穩定訓練"
echo "====================================================="
echo "技術細節："
echo "- Codebook dim: 512"
echo "- Transformer d_model: 128"
echo "- 使用 Linear(512→128) 投影層"
echo "- 混合 embedding 策略"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TTT_BATCH_SIZE=8
export TTT_NUM_WORKERS=2
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CONTENT_BATCHING=true
export CUDA_LAUNCH_BLOCKING=1
export EFFECTIVE_BATCH_SIZE=8

# 設置文件路徑
LOG_FILE="logs/transformer_codebook_${EXP_ID}.log"
OUTPUT_DIR="results/transformer_codebook_${EXP_ID}"

# 創建目錄
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# 自動選擇空閒的GPU (排除GPU 2)
echo "🔍 檢測可用的GPU..."
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
echo "GPU狀況:"
echo "$GPU_INFO"

# 只從GPU 0和1中選擇最空閒的
BEST_GPU=$(echo "$GPU_INFO" | awk -F',' '$1 == 0 || $1 == 1 {print $1, $2, $3}' | sort -k2,2n | head -1 | awk '{print $1}')

if [ -z "$BEST_GPU" ]; then
    echo "❌ 無法檢測到可用的GPU，使用預設GPU 1"
    export CUDA_VISIBLE_DEVICES=1
else
    export CUDA_VISIBLE_DEVICES=$BEST_GPU
    echo "✅ 選擇GPU $BEST_GPU"
fi

# 測試GPU和Codebook Embedding功能
echo "🧪 測試GPU和Codebook Embedding..."
python -c "
import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.WARNING)

print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Using GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name()}')
    
    # 測試 Codebook Embedding 功能
    try:
        from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser
        
        model = WavTokenizerTransformerDenoiser(
            config_path='config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
            model_path='models/wavtokenizer_large_speech_320_24k.ckpt',
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        print(f'✅ Codebook weights shape: {model.codebook_weights.shape}')
        print(f'✅ Codebook to d_model: {model.codebook_to_dmodel}')
        print(f'✅ Special token embedding: {model.special_token_embedding}')
        print('✅ Codebook Embedding 測試通過！')
        
    except Exception as e:
        print(f'❌ Codebook Embedding 測試失敗: {e}')
        exit(1)
else:
    print('❌ CUDA不可用！')
    exit(1)
" || { echo "❌ 測試失敗！"; exit 1; }

echo "運行環境設定:"
echo "- 架構: Codebook Embedding (預訓練)"
echo "- 批次大小: $TTT_BATCH_SIZE"
echo "- Transformer d_model: 128"
echo "- 日誌文件: $LOG_FILE"
echo "- 輸出目錄: $OUTPUT_DIR"
echo "====================================================="

# 激活conda環境
echo "激活 conda test 環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 清理CUDA緩存
echo "清理 CUDA 緩存..."
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 運行 Codebook Embedding 實驗
echo "🚀 開始 Codebook Embedding 訓練，時間: $(date)"
python wavtokenizer_transformer_denoising.py \
    --d_model 128 \
    --nhead 2 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --dim_feedforward 256 \
    --max_length 400 \
    --batch_size $TTT_BATCH_SIZE \
    --use_token_loss \
    --gradient_accumulation_steps 2 \
    --num_epochs 300 \
    --learning_rate 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 10 \
    --val_speakers girl9 boy7 \
    --train_speakers boy1 boy3 boy4 boy5 boy6 girl2 girl3 girl4 girl6 girl7 \
    2>&1 | tee -a $LOG_FILE

echo ""
echo "====================================================="
echo "Codebook Embedding 訓練完成，時間: $(date)"
echo "結果日誌保存在: $LOG_FILE"
echo "實驗結果保存在: $OUTPUT_DIR"

# 檢查是否成功完成
if [ -f "$OUTPUT_DIR/final_model.pth" ]; then
    echo "✅ 實驗成功完成！"
    echo ""
    echo "🎯 預期改進："
    echo "- 更快收斂（使用預訓練初始化）"
    echo "- 更好性能（保留音頻語義）"
    echo "- 更穩定訓練（一致的語義空間）"
    echo ""
    echo "📊 請查看訓練曲線對比："
    echo "- 舊版（隨機embedding）vs 新版（codebook embedding）"
else
    echo "⚠️ 實驗可能未完全完成，請檢查日誌"
fi

echo ""
echo "📝 更新實驗記錄到 REPORT.md..."
cat >> REPORT.md << EOF

---

## 🎯 Codebook Embedding 實驗 - ${TIMESTAMP}

### 實驗 ID: ${EXP_ID}

### 架構改進
- ✅ 使用 WavTokenizer 預訓練 Codebook 作為 Token Embedding
- ✅ 保留預訓練模型的語義結構
- ✅ 混合 embedding 策略：codebook tokens + special tokens

### 技術細節
- Codebook dim: 512
- Transformer d_model: 128
- Codebook tokens (0-4095): 使用預訓練 \`F.embedding(tokens, codebook_weights)\`
- Special tokens (pad/sos/eos): 使用可學習 \`nn.Embedding(3, 128)\`

### 訓練配置
- 批次大小: ${TTT_BATCH_SIZE}
- Epochs: 300
- Learning rate: 1e-4
- 訓練語者: boy1, boy3, boy4, boy5, boy6, girl2, girl3, girl4, girl6, girl7
- 驗證語者: girl9, boy7

### 預期結果
- 更快收斂速度
- 更好的最終性能
- 更穩定的訓練過程

### 結果
- 日誌: ${LOG_FILE}
- 輸出: ${OUTPUT_DIR}
- Git Commit: $(git rev-parse --short HEAD)

EOF

echo "✅ 記錄已更新到 REPORT.md"
echo ""
echo "🎉 Codebook Embedding 實驗執行完成！"
echo "======================================================"

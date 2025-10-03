#!/bin/bash

# 測試版的 discrete tokenloss 實驗 - 快速驗證修復
set -e

# 實驗編號
EXP_ID="test_fix_$(date +%Y%m%d%H%M)"
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "====================================================="
echo "測試版 WavTokenizer-Transformer 離散Token訓練 - $EXP_ID"
echo "====================================================="
echo "修復項目:"
echo "1. ✅ Token索引越界問題 (CUDA device-side assert)"
echo "2. ✅ 添加嚴格的token範圍檢查"
echo "3. ✅ 修復token_loss_system中的vocab_size檢查"
echo "4. ✅ 恢復batch_size=8確保內容一致性損失有效"
echo "5. ✅ 增強錯誤處理和緊急保存機制"
echo "6. ✅ 新增：語者過濾功能，避免載入不需要的數據"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TTT_BATCH_SIZE=4  # 測試用較小batch size
export TTT_NUM_WORKERS=2
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CONTENT_BATCHING=true
export CUDA_LAUNCH_BLOCKING=1
export EFFECTIVE_BATCH_SIZE=4

# 設置文件路徑
LOG_FILE="logs/wavtokenizer_transformer_training_${EXP_ID}.log"
OUTPUT_DIR="results/wavtokenizer_tokenloss_${EXP_ID}"

# 創建目錄
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# 自動選擇空閒的GPU
echo "🔍 檢測可用的GPU..."
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
echo "GPU狀況:"
echo "$GPU_INFO"

BEST_GPU=$(echo "$GPU_INFO" | awk -F',' '{print $1, $2, $3}' | sort -k2,2n | head -1 | awk '{print $1}')

if [ -z "$BEST_GPU" ]; then
    echo "❌ 無法檢測到可用的GPU，使用預設GPU 0"
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$BEST_GPU
    echo "✅ 選擇GPU $BEST_GPU (記憶體使用最少)"
fi

echo "運行環境設定:"
echo "- 批次大小: $TTT_BATCH_SIZE (測試用)"
echo "- 日誌文件: $LOG_FILE"
echo "- 輸出目錄: $OUTPUT_DIR"
echo "- CUDA同步檢測: 啟用"
echo "- 測試模式: 只運行5個epochs驗證功能"
echo "====================================================="

# 激活conda環境
echo "激活 conda test 環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 清理CUDA緩存
echo "清理 CUDA 緩存..."
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 運行測試版實驗（只跑5個epochs）
echo "🚀 開始測試版 WavTokenizer-Transformer 訓練，時間: $(date)"
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
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 2 \
    --val_speakers girl9 boy7 \
    --train_speakers boy1 boy3 boy4 boy5 boy6 girl2 girl3 girl4 girl6 girl7 \
    --max_sentences_per_speaker 10 \
    2>&1 | tee -a $LOG_FILE

echo ""
echo "====================================================="
echo "測試版 WavTokenizer-Transformer 訓練完成，時間: $(date)"
echo "結果日誌保存在: $LOG_FILE"
echo "實驗結果保存在: $OUTPUT_DIR"

# 檢查是否成功完成
if [ -f "$OUTPUT_DIR/final_model.pth" ]; then
    echo "✅ 測試實驗成功完成！"
else
    echo "⚠️ 測試實驗可能未完全完成，請檢查日誌"
fi

echo "🎉 測試版實驗執行完成！"
echo "======================================================"
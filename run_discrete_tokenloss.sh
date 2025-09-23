#!/bin/bash

# 確保腳本在錯誤時會停止運行
set -e

# 獲取當前日期時間作為實驗編號
EXP_ID=$(date +%Y%m%d%H%M)
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 顯示腳本運行訊息
echo "====================================================="
echo "開始執行 WavTokenizer-Transformer 離散Token訓練 - $EXP_ID"
echo "====================================================="
echo "模型: WavTokenizer-Transformer (離散Token空間降噪)"
echo "輸出目錄: results/wavtokenizer_tokenloss_${EXP_ID}"
echo ""
echo "🔧 架構特點:"
echo "1. ✅ 離散Token空間降噪 (vs ttt2.py 連續特徵)"
echo "2. ✅ Transformer架構 (vs ttt2.py ResidualBlock)"
echo "3. ✅ Token Loss系統: 移植ttt2.py損失邏輯到離散空間"
echo "4. ✅ 輕量化設計: 減少記憶體使用"
echo "5. ✅ 僅使用 box 材質數據進行訓練"
echo ""
echo "🎯 訓練參數 (參考 run_fixed_ttt2_branch.sh):"
echo "1. --use_token_loss: 使用Token Loss系統而非純CrossEntropy"
echo "2. 輕量化Transformer: d_model=256, 3層encoder/decoder"
echo "3. ONLY_USE_BOX_MATERIAL=true: 僅使用 box 材質數據"
echo "4. batch_size=8: 確保內容一致性損失計算 (與ttt2.py一致)"
echo "====================================================="

# 設置環境變數 (完全參考 run_fixed_ttt2_branch.sh)
export ONLY_USE_BOX_MATERIAL=true     # 僅處理 box 材質
export PYTHONUNBUFFERED=1           # 即時輸出日誌，不進行緩衝
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 限制CUDA記憶體分配塊大小
export TTT_BATCH_SIZE=8             # 增大批次大小以確保每個批次有相同內容ID的多個樣本
export TTT_NUM_WORKERS=4            # 資料載入工作線程數
export TTT_EXPERIMENT_ID="${EXP_ID}" # 設置實驗ID
export INPUT_SAMPLE_RATE=16000      # 設置輸入音頻採樣率
export CONTENT_BATCHING=true        # 啟用內容感知批次採樣，確保相同內容的樣本在同一批次

# 設置運行參數
LOG_FILE="logs/wavtokenizer_transformer_training_${EXP_ID}.log"  # 日誌文件路徑
OUTPUT_DIR="results/wavtokenizer_tokenloss_${EXP_ID}"

# 創建日誌目錄
mkdir -p logs

echo "運行環境設定:"
echo "- 批次大小: $TTT_BATCH_SIZE (內容一致性損失需要)"
echo "- 資料載入線程數: $TTT_NUM_WORKERS"
echo "- 日誌文件: $LOG_FILE"
echo "- 輸出目錄: $OUTPUT_DIR"
echo "- 僅使用BOX材質: $ONLY_USE_BOX_MATERIAL"
echo "====================================================="

# 激活 conda 環境
echo "激活 conda test 環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 運行前清理CUDA緩存
echo "清理 CUDA 緩存..."
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 運行模型，同時將輸出導向至終端和日誌文件
echo "🚀 開始 WavTokenizer-Transformer 訓練，時間: $(date)"
echo "使用輕量化 Transformer + Token Loss 系統..."
python wavtokenizer_transformer_denoising.py \
    --d_model 256 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --dim_feedforward 1024 \
    --max_length 256 \
    --batch_size 8 \
    --use_token_loss \
    --gradient_accumulation_steps 2 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 25 \
    --val_speakers girl9 boy7 \
    --max_sentences_per_speaker 100 \
    2>&1 | tee -a $LOG_FILE

# 顯示完成訊息
echo ""
echo "====================================================="
echo "WavTokenizer-Transformer 訓練完成，時間: $(date)"
echo "結果日誌保存在: $LOG_FILE"
echo "實驗結果保存在: $OUTPUT_DIR"

# 自動更新實驗報告 (參考 run_fixed_ttt2_branch.sh 格式)
echo "" >> $REPORT_FILE
echo "## WavTokenizer-Transformer 離散Token訓練 - TOKEN_$EXP_ID" >> $REPORT_FILE
echo "**執行時間:** $TIMESTAMP" >> $REPORT_FILE
echo "**模式:** 輕量化Transformer + Token Loss" >> $REPORT_FILE
echo "**輸出目錄:** $OUTPUT_DIR" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🔧 關鍵特色" >> $REPORT_FILE
echo "1. **離散Token空間:** vs ttt2.py連續特徵空間" >> $REPORT_FILE
echo "2. **Transformer架構:** vs ttt2.py ResidualBlock架構" >> $REPORT_FILE
echo "3. **Token Loss系統:** 移植ttt2.py損失邏輯到離散空間" >> $REPORT_FILE
echo "4. **內容一致性損失:** batch_size=8 確保相同內容ID樣本" >> $REPORT_FILE
echo "5. **數據一致性:** 與ttt2.py相同的BOX材質、語者分組" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🎯 訓練設定" >> $REPORT_FILE
echo "- **模型:** WavTokenizer-Transformer (輕量化)" >> $REPORT_FILE
echo "- **架構:** d_model=256, 3+3層, nhead=4" >> $REPORT_FILE
echo "- **損失函數:** Token Loss系統 (ttt2.py移植版)" >> $REPORT_FILE
echo "- **材質:** 僅 box 材質 (與ttt2.py一致)" >> $REPORT_FILE
echo "- **批次大小:** $TTT_BATCH_SIZE (內容一致性損失需要)" >> $REPORT_FILE
echo "- **日誌檔案:** \`$LOG_FILE\`" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 📊 預期對比" >> $REPORT_FILE
echo "- 對比ttt2.py: 離散 vs 連續特徵空間處理效果" >> $REPORT_FILE
echo "- 對比架構: Transformer vs ResidualBlock 降噪能力" >> $REPORT_FILE
echo "- 對比損失: Token Loss在離散空間的適應性" >> $REPORT_FILE
echo "- 對比記憶體: 輕量化設計的效率提升" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "----" >> $REPORT_FILE

echo "已更新實驗報告: $REPORT_FILE"
echo "🎉 WavTokenizer-Transformer 訓練啟動完成！"
echo "======================================================"

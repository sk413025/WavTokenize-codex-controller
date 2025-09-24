#!/bin/bash

# 確保腳本在錯誤時會停止運行
set -e

# 獲取當前日期時間作為實驗編號
EXP_ID=$(date +%Y%m%d%H%M)
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 顯示腳本運行訊息
echo "====================================================="
echo "開始執行 WavTokenizer-Transformer 純交叉熵實驗 - $EXP_ID"
echo "====================================================="
echo "實驗類型: 純交叉熵損失 (Cross-Entropy Only)"
echo "輸出目錄: results/crossentropy_exp_${EXP_ID}"
echo ""
echo "🎯 實驗目的:"
echo "1. ✅ 驗證純交叉熵能否實現語者風格還原和降噪"
echo "2. ✅ 對比 Token Loss vs 純交叉熵的性能差異"
echo "3. ✅ 測試離散Token空間的基礎學習能力"
echo "4. ✅ 評估交叉熵在音頻降噪任務的有效性"
echo ""
echo "🔧 關鍵配置:"
echo "1. 禁用 --use_token_loss: 僅使用純交叉熵損失"
echo "2. 超輕量化Transformer: d_model=128, 2層encoder/decoder, ff=256, heads=2"
echo "3. 相同語者分割: 訓練集[boy1,3,4,5,6+girl2,3,4,6,7] vs 驗證集[girl9,boy7]"
echo "4. 相同數據集: box材質，確保與Token Loss實驗對比公平"
echo "5. 快速驗證: 減少訓練輪數，專注於損失函數對比"
echo ""
echo "📊 預期結果:"
echo "• 交叉熵損失曲線應該穩定下降"
echo "• 模型應該學會 noisy_tokens → clean_tokens 映射"
echo "• 還原音頻應該保持語者特徵並實現降噪"
echo "• 與Token Loss結果對比分析兩種方法優劣"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true     # 僅處理 box 材質
export PYTHONUNBUFFERED=1           # 即時輸出日誌，不進行緩衝
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 限制CUDA記憶體分配塊大小
export TTT_BATCH_SIZE=8             # 批次大小
export TTT_NUM_WORKERS=4            # 資料載入工作線程數
export TTT_EXPERIMENT_ID="${EXP_ID}" # 設置實驗ID
export INPUT_SAMPLE_RATE=16000      # 設置輸入音頻採樣率
export CONTENT_BATCHING=true        # 啟用內容感知批次採樣

# 設置運行參數
LOG_FILE="logs/crossentropy_experiment_${EXP_ID}.log"
OUTPUT_DIR="results/crossentropy_exp_${EXP_ID}"

# 創建必要目錄
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "運行環境設定:"
echo "- 實驗類型 \"Experiment Type\": 純交叉熵實驗"
echo "- 批次大小 \"Batch Size\": $TTT_BATCH_SIZE"
echo "- 資料載入線程數: $TTT_NUM_WORKERS"
echo "- 日誌文件: $LOG_FILE"
echo "- 輸出目錄: $OUTPUT_DIR"
echo "- 僅使用 box 材質: $ONLY_USE_BOX_MATERIAL"
echo "====================================================="

# 激活 conda 環境
echo "激活 conda test 環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 運行前清理CUDA緩存
echo "清理 CUDA 緩存..."
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 執行純交叉熵實驗
echo "🚀 開始純交叉熵實驗，時間: $(date)"
echo "重要: 不使用 --use_token_loss 參數，僅使用標準交叉熵損失"

python wavtokenizer_transformer_denoising.py \
    --d_model 128 \
    --nhead 2 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --dim_feedforward 256 \
    --max_length 256 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 10 \
    --val_speakers girl9 boy7 \
    --train_speakers boy1 boy3 boy4 boy5 boy6 girl2 girl3 girl4 girl6 girl7 \
    --max_sentences_per_speaker 100 \
    2>&1 | tee -a $LOG_FILE

# 顯示完成訊息
echo ""
echo "====================================================="
echo "純交叉熵實驗執行完成，時間: $(date)"
echo "結果日誌保存在: $LOG_FILE"
echo "實驗結果保存在: $OUTPUT_DIR"
echo "====================================================="

# 自動更新實驗報告
echo "" >> $REPORT_FILE
echo "## 純交叉熵音頻降噪實驗 - CROSSENTROPY_EXP_$EXP_ID" >> $REPORT_FILE
echo "**執行時間:** $TIMESTAMP" >> $REPORT_FILE
echo "**實驗類型:** 純交叉熵損失驗證實驗" >> $REPORT_FILE
echo "**輸出目錄:** $OUTPUT_DIR" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🎯 實驗目的與背景" >> $REPORT_FILE
echo "1. **驗證假設:** 純交叉熵損失能否實現語者風格還原和降噪" >> $REPORT_FILE
echo "2. **對比分析:** 與 Token Loss 系統的性能差異比較" >> $REPORT_FILE
echo "3. **機制探索:** 離散 Token 空間中交叉熵的學習機制" >> $REPORT_FILE
echo "4. **基準建立:** 為後續複雜損失函數提供基準線" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🔧 實驗配置" >> $REPORT_FILE
echo "- **損失函數:** 純交叉熵 (禁用 Token Loss)" >> $REPORT_FILE
echo "- **架構:** 超輕量化 Transformer (d_model=128, layers=2+2)" >> $REPORT_FILE
echo "- **語者分割:** 訓練集 10 人，驗證集 2 人 (與 Token Loss 實驗一致)" >> $REPORT_FILE
echo "- **訓練輪數:** 50 epochs (快速驗證)" >> $REPORT_FILE
echo "- **數據集:** 僅 box 材質音頻數據" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 📊 預期分析指標" >> $REPORT_FILE
echo "1. **損失收斂:** 交叉熵損失的下降趨勢和穩定性" >> $REPORT_FILE
echo "2. **語者還原:** 還原音頻是否保持原語者特徵" >> $REPORT_FILE
echo "3. **降噪效果:** 噪聲去除程度和音質改善" >> $REPORT_FILE
echo "4. **vs Token Loss:** 兩種方法的性能對比分析" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🔍 實驗結果" >> $REPORT_FILE
echo "*實驗完成後填入具體結果和分析*" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "---" >> $REPORT_FILE

echo ""
echo "🎉 交叉熵實驗腳本執行完成!"
echo "📋 實驗記錄已自動添加到 $REPORT_FILE"
echo "🔍 可使用 'tail -f $LOG_FILE' 監控訓練進度"
echo ""
echo "🆚 對比實驗建議:"
echo "1. 先執行此純交叉熵實驗"
echo "2. 再執行 run_discrete_tokenloss.sh (Token Loss實驗)"
echo "3. 對比兩種方法的損失曲線和音頻質量"
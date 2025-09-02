#!/bin/bash

# 確保腳本在錯誤時會停止運行
set -e

# 獲取當前日期時間作為實驗編號
EXP_ID=$(date +%Y%m%d%H%M)
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 顯示腳本運行訊息
echo "====================================================="
echo "開始執行實驗方案一：階層式內容一致性損失 - $EXP_ID"
echo "====================================================="
echo "實驗分支: experiment-hierarchical-content"
echo "輸出目錄: results/tsne_outputs/exp1-hierarchical-$EXP_ID"
echo ""
echo "🧪 實驗方案一："
echo "1. ✅ 階層式內容一致性損失：結合連續和離散特徵"
echo "2. ✅ 連續特徵權重: 0.7（可調整）"
echo "3. ✅ 離散特徵權重: 0.3（可調整）"
echo "4. ✅ 內容一致性損失權重: 0.01"
echo "5. ✅ 每位與者限制: 前100句話"
echo ""
echo "🎯 實驗參數:"
echo "1. --experiment_hierarchical_content: 啟用階層式內容一致性損失"
echo "2. --hierarchy_alpha 0.7: 連續特徵權重"
echo "3. --content_alpha 0.01: 內容一致性損失權重"
echo "4. ONLY_USE_BOX_MATERIAL=true: 僅使用 box 材質數據進行訓練"
echo "5. 每位與者限制使用前100句話"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true     # 僅處理 box 材質
export PYTHONUNBUFFERED=1           # 即時輸出日誌，不進行緩衝
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 限制CUDA記憶體分配塊大小
export TTT_BATCH_SIZE=8             # 增大批次大小以確保每個批次有相同內容ID的多個樣本
export TTT_NUM_WORKERS=4            # 資料載入工作線程數
export TTT_EXPERIMENT_ID="${EXP_ID}" # 設置實驗ID
export INPUT_SAMPLE_RATE=16000      # 設置輸入音頻採樣率
export CONTENT_BATCHING=true        # 啟用內容感知批次採樣，確保相同內容的樣本在同一批次

# 設置運行參數
LOG_FILE="logs/exp1_hierarchical_content_${EXP_ID}.log"  # 實驗專用日誌文件路徑
OUTPUT_DIR="results/tsne_outputs/exp1-hierarchical-$EXP_ID"

# 創建日誌目錄
mkdir -p logs

echo "運行環境設定:"
echo "- 批次大小 \"Batch Size\": $TTT_BATCH_SIZE"
echo "- 資料載入線程數: $TTT_NUM_WORKERS"
echo "- 日誌文件: $LOG_FILE"
echo "- 實驗輸出目錄: $OUTPUT_DIR"
echo "====================================================="

# 激活 conda 環境
echo "激活 conda test 環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 運行前清理CUDA緩存
echo "清理 CUDA 緩存..."
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 運行模型，同時將輸出導向至終端和日誌文件
echo "🚀 開始實驗方案一訓練，時間: $(date)"
echo "使用階層式內容一致性損失（連續+離散特徵）..."
python ttt2.py \
    --experiment_hierarchical_content \
    --hierarchy_alpha 0.7 \
    --content_alpha 0.01 \
    2>&1 | tee -a $LOG_FILE

# 顯示完成訊息
echo ""
echo "====================================================="
echo "實驗方案一程序執行完成，時間: $(date)"
echo "結果日誌保存在: $LOG_FILE"
echo "實驗結果保存在: $OUTPUT_DIR"

# 自動更新實驗報告
echo "" >> $REPORT_FILE
echo "## 實驗方案一：階層式內容一致性損失 - EXP1_$EXP_ID" >> $REPORT_FILE
echo "**執行時間:** $TIMESTAMP" >> $REPORT_FILE
echo "**實驗類型:** 階層式內容一致性損失（連續+離散特徵）" >> $REPORT_FILE
echo "**輸出目錄:** $OUTPUT_DIR" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🧪 實驗設計" >> $REPORT_FILE
echo "1. **階層式損失:** 結合連續特徵（中間層embedding）和離散特徵（codebook index）" >> $REPORT_FILE
echo "2. **連續特徵權重:** 0.7（保留豐富的韻律資訊）" >> $REPORT_FILE
echo "3. **離散特徵權重:** 0.3（強化語意一致性）" >> $REPORT_FILE
echo "4. **內容一致性權重:** 0.01" >> $REPORT_FILE
echo "5. **數據限制:** 每位與者限制前100句話" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🎯 實驗目標" >> $REPORT_FILE
echo "- 驗證階層式內容一致性損失是否能同時保留韻律細節和語意一致性" >> $REPORT_FILE
echo "- 評估連續和離散特徵結合的效果" >> $REPORT_FILE
echo "- 觀察語音品質和內容保留的平衡" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 📊 評估指標" >> $REPORT_FILE
echo "- 語音品質（主觀評估）" >> $REPORT_FILE
echo "- 內容保留度（ASR準確率）" >> $REPORT_FILE
echo "- 韻律相似度" >> $REPORT_FILE
echo "- 訓練穩定性（損失收斂）" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 📁 輸出文件" >> $REPORT_FILE
echo "- **日誌檔案:** \`$LOG_FILE\`" >> $REPORT_FILE
echo "- **模型檔案:** \`$OUTPUT_DIR/models/\`" >> $REPORT_FILE
echo "- **特徵檔案:** \`$OUTPUT_DIR/features/\`" >> $REPORT_FILE
echo "- **t-SNE圖表:** \`$OUTPUT_DIR/tsne_plots/\`" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "----" >> $REPORT_FILE

echo "已更新實驗報告: $REPORT_FILE"
echo "🎉 實驗方案一訓練啟動完成！"
echo "======================================================"

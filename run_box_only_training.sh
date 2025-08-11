#!/bin/bash

# 確保腳本在錯誤時會停止運行
set -e

# 獲取當前日期時間作為實驗編號
EXP_ID=$(date +%Y%m%d%H%M)
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 顯示腳本運行訊息
echo "====================================================="
echo "開始執行 TTT 模型訓練 - $EXP_ID"
echo "====================================================="
echo "模型特點: 使用嚴格分層損失 + 增強版內容一致性損失 + 僅使用 box 材質"
echo ""
echo "參數設定:"
echo "1. --tsne_flow_with_content: 處理流程與tsne.py保持一致，但同時計算內容一致性損失"
echo "2. --use_layered_loss: 使用分層損失機制"
echo "3. --first_two_blocks_only: 嚴格分層損失設計:"
echo "   - 中間層: 完全自由學習，無損失監督"
echo "   - 第二層: 僅計算內容一致性損失 \"確保語句結構保留\""
echo "   - 最後層: 僅計算L2特徵損失 \"確保降噪效果\""
echo "4. ONLY_USE_BOX_MATERIAL=true: 僅使用 box 材質數據進行訓練"
echo "5. 【改進】增強版內容一致性損失計算: 結合方向相似度\"餘弦距離\"與分布差異\"L2距離\""
echo "   - 方向相似度: 保持語義內容一致性"
echo "   - 分布差異: 維持語句結構與語調變化"
echo "   - 兩者結合: 更好地保留語句的整體特性"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true     # 僅處理 box 材質
export PYTHONUNBUFFERED=1           # 即時輸出日誌，不進行緩衝
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 限制CUDA記憶體分配塊大小
export TTT_BATCH_SIZE=4             # 較小批次大小以節省記憶體
export TTT_NUM_WORKERS=4            # 資料載入工作線程數
export TTT_EXPERIMENT_ID="${EXP_ID}" # 設置實驗ID

# 設置運行參數
LOG_FILE="logs/ttt_training_${EXP_ID}.log"  # 日誌文件路徑

# 創建日誌目錄
mkdir -p logs

echo "運行環境設定:"
echo "- 批次大小 \"Batch Size\": $TTT_BATCH_SIZE"
echo "- 資料載入線程數: $TTT_NUM_WORKERS"
echo "- 日誌文件: $LOG_FILE"
echo "====================================================="

# 運行前清理CUDA緩存
echo "清理 CUDA 緩存..."
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 運行模型，同時將輸出導向至終端和日誌文件
echo "開始模型訓練，時間: $(date)"
python ttt.py \
    --tsne_flow_with_content \
    --use_layered_loss \
    --first_two_blocks_only \
    \
    2>&1 | tee -a $LOG_FILE

# 顯示完成訊息
echo ""
echo "====================================================="
echo "程序執行完成，時間: $(date)"
echo "結果日誌保存在: $LOG_FILE"

# 自動更新實驗報告
echo "" >> $REPORT_FILE
echo "## TTT模型訓練 - $EXP_ID" >> $REPORT_FILE
echo "**執行時間:** $TIMESTAMP" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 訓練設定" >> $REPORT_FILE
echo "- **模型:** TTT (改進版內容一致性損失)" >> $REPORT_FILE
echo "- **損失函數:** 嚴格分層損失 - 倒數第二層內容損失 + 最後層L2損失" >> $REPORT_FILE
echo "- **材質:** 僅 box 材質" >> $REPORT_FILE
echo "- **批次大小:** $TTT_BATCH_SIZE" >> $REPORT_FILE
echo "- **日誌檔案:** \`$LOG_FILE\`" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 損失計算特點" >> $REPORT_FILE
echo "- 嚴格分層損失設計，不同於en_train.py的實現方式" >> $REPORT_FILE
echo "- 改進的內容一致性損失，結合方向相似度與分布差異，更好地保留語句結構" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "----" >> $REPORT_FILE

echo "已更新實驗報告: $REPORT_FILE"
echo "======================================================"

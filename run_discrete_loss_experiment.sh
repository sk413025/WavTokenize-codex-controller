#!/bin/bash

# 確保腳本在錯誤時會停止運行
set -e

# 顯示腳本運行訊息
echo "====================================================="
echo "開始執行 TTT3 模型 (離散編碼損失實驗) - 簡化版"
echo "====================================================="
echo "參數設定:"
echo "1. --use_discrete_loss: 啟用離散編碼的L2和內容一致性損失，結合連續特徵損失"
echo "====================================================="

# 設置日期和實驗標識符
DATE=$(date +"%Y%m%d")
EXP_ID="EXP07_DISCRETE_LOSS"
LOG_DIR="/home/sbplab/ruizi/WavTokenize/logs"

# 創建日誌目錄
mkdir -p $LOG_DIR

# 日誌文件路徑
LOG_FILE="$LOG_DIR/${EXP_ID}_${DATE}.log"

# 運行前清理CUDA緩存
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 執行 ttt3_simplified.py 使用離散編碼損失
echo "開始執行離散編碼損失實驗 - $(date)" | tee $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# 使用簡化版腳本運行
python ttt3_simplified.py --use_discrete_loss 2>&1 | tee -a $LOG_FILE

# 檢查是否成功
if [ $? -eq 0 ]; then
    echo "離散編碼損失實驗完成！" | tee -a $LOG_FILE
else
    echo "離散編碼損失實驗執行失敗。" | tee -a $LOG_FILE
    exit 1
fi

# 顯示完成訊息
echo ""
echo "====================================================="
echo "程序執行完成 - $(date)"
echo "日誌文件位於: $LOG_FILE"
echo "====================================================="

#!/bin/bash
# 執行離散特徵語義解離實驗和噪聲抑制實驗

# 設定日期和實驗標識符
DATE=$(date +"%Y%m%d")
EXP_ID="EXP_DISCRETE"
LOG_DIR="/home/sbplab/ruizi/WavTokenize/logs"

# 創建日誌目錄
mkdir -p $LOG_DIR

# 日誌文件路徑
LOG_FILE="$LOG_DIR/${EXP_ID}_${DATE}.log"

# 打印開始信息
echo "開始執行離散特徵語義解離實驗和噪聲抑制實驗 - $(date)" | tee $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Step 1: 首先執行離散特徵語義解離實驗，分析各層的語義含義
echo "執行離散特徵語義解離實驗..." | tee -a $LOG_FILE
python exp_discrete_swap.py 2>&1 | tee -a $LOG_FILE

# 檢查是否成功
if [ $? -eq 0 ]; then
    echo "離散特徵語義解離實驗完成！" | tee -a $LOG_FILE
else
    echo "離散特徵語義解離實驗失敗，退出執行。" | tee -a $LOG_FILE
    exit 1
fi

# 等待 10 秒，確保文件寫入完成
sleep 10

# Step 2: 根據解離實驗結果，執行噪聲抑制實驗
echo "----------------------------------------" | tee -a $LOG_FILE
echo "執行離散特徵噪聲抑制實驗..." | tee -a $LOG_FILE
python exp_noise_reduction.py 2>&1 | tee -a $LOG_FILE

# 檢查是否成功
if [ $? -eq 0 ]; then
    echo "離散特徵噪聲抑制實驗完成！" | tee -a $LOG_FILE
else
    echo "離散特徵噪聲抑制實驗失敗。" | tee -a $LOG_FILE
    exit 1
fi

# 打印完成信息
echo "----------------------------------------" | tee -a $LOG_FILE
echo "所有實驗已完成 - $(date)" | tee -a $LOG_FILE
echo "日誌文件位於: $LOG_FILE" | tee -a $LOG_FILE

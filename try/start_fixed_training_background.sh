#!/bin/bash

# 在背景執行修復後的訓練腳本

cd "$(dirname "$0")"

echo "======================================"
echo "在背景啟動修復後的訓練"
echo "======================================"

# 執行訓練並記錄 PID
nohup bash run_fixed_training.sh > /dev/null 2>&1 &
TRAIN_PID=$!

echo "✅ 訓練已在背景啟動"
echo "PID: $TRAIN_PID"
echo ""
echo "監控指令："
echo "  - 查看日誌: tail -f ../logs/token_denoising_fixed_*.log"
echo "  - 查看進程: ps aux | grep $TRAIN_PID"
echo "  - 停止訓練: kill $TRAIN_PID"
echo ""
echo "實驗 ID 會在日誌檔名中"
echo "======================================"

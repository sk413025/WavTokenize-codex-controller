#!/bin/bash
# 監控訓練到 Epoch 2

LOG_FILE="logs/large_tokenloss_FINAL_20251021_044006.log"

echo "================================================"
echo "訓練監控 - 目標: Epoch 2"
echo "日誌文件: $LOG_FILE"
echo "開始時間: $(date)"
echo "================================================"
echo ""

# 檢查當前進度
while true; do
    CURRENT=$(tail -50 "$LOG_FILE" 2>/dev/null | grep "Epoch [0-9]" | tail -1)
    
    # 檢查進程
    if ! ps aux | grep -q "[p]ython.*wavtokenizer"; then
        echo "❌ 訓練進程已停止"
        tail -30 "$LOG_FILE" | grep -E "ERROR|Traceback"
        break
    fi
    
    # 顯示當前進度
    echo "$(date +%H:%M:%S) - $CURRENT" | grep -o "Epoch [0-9]*.*Acc=[0-9.]*%"
    
    # 檢查是否達到 Epoch 2
    if echo "$CURRENT" | grep -q "Epoch 2"; then
        echo ""
        echo "================================================"
        echo "✅ 已達到 Epoch 2！"
        echo "結束時間: $(date)"
        echo "================================================"
        echo ""
        
        # 顯示統計
        echo "【訓練統計】"
        grep "Epoch 1.*Token Loss.*100%" "$LOG_FILE" | tail -1
        grep "Epoch 2" "$LOG_FILE" | head -3
        
        echo ""
        echo "【錯誤檢查】"
        ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE" || echo 0)
        CUDA_ERROR=$(grep -c "CUDA error" "$LOG_FILE" || echo 0)
        echo "ERROR 數量: $ERROR_COUNT"
        echo "CUDA error 數量: $CUDA_ERROR"
        
        if [ "$ERROR_COUNT" -eq 0 ] && [ "$CUDA_ERROR" -eq 0 ]; then
            echo "✅ 沒有錯誤，訓練正常"
        fi
        
        break
    fi
    
    sleep 30
done

echo ""
echo "監控結束"

#!/bin/bash
# 監控 Exp K 完成後自動啟動 Exp M

echo "=========================================="
echo "Monitor: Exp K → Exp M"
echo "=========================================="
echo "Started at: $(date)"
echo ""

LOG_FILE="/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/exp_k.log"

# 等待 Exp K 完成
while true; do
    if grep -q "訓練完成!" "$LOG_FILE" 2>/dev/null || grep -q "Training completed!" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "[$(date)] Exp K completed!"
        break
    fi

    if grep -q "Error\|Exception\|Traceback" "$LOG_FILE" 2>/dev/null; then
        # 檢查是否是真正的錯誤（排除 warning）
        if tail -50 "$LOG_FILE" 2>/dev/null | grep -q "RuntimeError\|CUDA out of memory\|KeyboardInterrupt"; then
            echo ""
            echo "[$(date)] Exp K encountered an error!"
            tail -20 "$LOG_FILE"
            exit 1
        fi
    fi

    # 顯示當前進度
    CURRENT=$(tail -10 "$LOG_FILE" 2>/dev/null | grep -oP "Epoch \d+/\d+" | tail -1)
    if [ -n "$CURRENT" ]; then
        echo -ne "\r[$(date '+%H:%M:%S')] Exp K running: $CURRENT    "
    fi

    sleep 60
done

echo ""
echo "=========================================="
echo "Starting Exp M..."
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-feature-analysis
nohup bash exp_0113/run_exp_m.sh > exp_0113/exp_m.log 2>&1 &

echo "Exp M started! PID: $!"
echo "Log: exp_0113/exp_m.log"
echo ""
echo "Finished at: $(date)"

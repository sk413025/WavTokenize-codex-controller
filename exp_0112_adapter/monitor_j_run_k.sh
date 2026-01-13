#!/bin/bash
# 監控 Exp J 完成後自動啟動 Exp K

echo "=========================================="
echo "Monitor: Exp J → Exp K"
echo "=========================================="
echo "Started at: $(date)"
echo ""

LOG_FILE="/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_adapter/exp_j.log"

# 等待 Exp J 完成
while true; do
    if grep -q "Training completed!" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "[$(date)] Exp J completed!"
        break
    fi

    if grep -q "Error\|Exception\|Traceback" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "[$(date)] Exp J encountered an error!"
        tail -20 "$LOG_FILE"
        exit 1
    fi

    # 顯示當前進度
    CURRENT=$(tail -5 "$LOG_FILE" 2>/dev/null | grep -oP "Epoch \d+/\d+" | tail -1)
    if [ -n "$CURRENT" ]; then
        echo -ne "\r[$(date '+%H:%M:%S')] Exp J running: $CURRENT    "
    fi

    sleep 60
done

echo ""
echo "=========================================="
echo "Starting Exp K..."
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-feature-analysis
nohup ./exp_0112_intermediate/run_exp_k.sh > exp_0112_intermediate/exp_k.log 2>&1 &

echo "Exp K started! PID: $!"
echo "Log: exp_0112_intermediate/exp_k.log"
echo ""
echo "Finished at: $(date)"

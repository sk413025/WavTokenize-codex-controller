#!/bin/bash
# 監控 Exp K v2 完成後自動執行 r1024
#
# 使用方式:
#   nohup bash exp_test/monitor_k_run_r1024.sh > exp_test/monitor_r1024.log 2>&1 &

set -e

echo "=========================================="
echo "監控 Exp K v2 並等待完成..."
echo "=========================================="
echo "開始時間: $(date)"
echo ""

# 尋找 Exp K v2 的進程
check_exp_k() {
    pgrep -f "train_v2.py" > /dev/null 2>&1
    return $?
}

# 等待 Exp K v2 完成
if check_exp_k; then
    echo "偵測到 Exp K v2 正在運行，開始監控..."
    while check_exp_k; do
        echo "[$(date '+%H:%M:%S')] Exp K v2 仍在運行中..."
        sleep 60  # 每分鐘檢查一次
    done
    echo ""
    echo "=========================================="
    echo "Exp K v2 已完成! 時間: $(date)"
    echo "=========================================="
else
    echo "未偵測到 Exp K v2 進程，直接開始 r1024..."
fi

# 等待 10 秒確保資源釋放
echo "等待 10 秒讓 GPU 記憶體釋放..."
sleep 10

# 開始執行 r1024
echo ""
echo "=========================================="
echo "開始執行 exp_test r1024..."
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-feature-analysis
bash exp_test/run_r1024.sh

echo ""
echo "=========================================="
echo "r1024 完成! 時間: $(date)"
echo "=========================================="

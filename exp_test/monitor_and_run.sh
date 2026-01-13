#!/bin/bash
# 監控 Exp L 完成後自動執行 exp_test
#
# 使用方式:
#   nohup bash exp_test/monitor_and_run.sh > exp_test/monitor.log 2>&1 &

set -e

echo "=========================================="
echo "監控 Exp L 並等待完成..."
echo "=========================================="
echo "開始時間: $(date)"
echo ""

# 尋找 Exp L 的進程
check_exp_l() {
    pgrep -f "train_exp_l.py" > /dev/null 2>&1
    return $?
}

# 等待 Exp L 完成
if check_exp_l; then
    echo "偵測到 Exp L 正在運行，開始監控..."
    while check_exp_l; do
        echo "[$(date '+%H:%M:%S')] Exp L 仍在運行中..."
        sleep 60  # 每分鐘檢查一次
    done
    echo ""
    echo "=========================================="
    echo "Exp L 已完成! 時間: $(date)"
    echo "=========================================="
else
    echo "未偵測到 Exp L 進程，直接開始 exp_test..."
fi

# 等待 5 秒確保資源釋放
echo "等待 5 秒讓 GPU 記憶體釋放..."
sleep 5

# 開始執行 exp_test
echo ""
echo "=========================================="
echo "開始執行 exp_test..."
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-feature-analysis
bash exp_test/run_all.sh

echo ""
echo "=========================================="
echo "所有實驗完成! 時間: $(date)"
echo "=========================================="

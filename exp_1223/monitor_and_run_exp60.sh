#!/bin/bash
# 監控腳本: 等待 exp_1222 完成後自動運行 Exp60
#
# 原理:
# 1. 檢測 exp_1222/run_all_sequential.sh 的進程是否存在
# 2. 當進程結束後，自動啟動 exp_1223/run_exp60_speaker_film.sh
#
# 使用方式:
#   nohup bash exp_1223/monitor_and_run_exp60.sh > exp_1223/monitor.log 2>&1 &
#
# 注意:
# - 此腳本假設 exp_1222 正在運行中
# - 預設使用 GPU 0 (與 exp_1222 相同)
# - 可透過參數選擇運行 FiLM 或 CrossAttn 版本

set -e

# 參數: film (預設) 或 crossattn
EXP60_TYPE="${1:-film}"

echo "=============================================="
echo "Monitor Script: Wait for exp_1222 → Run Exp60"
echo "=============================================="
echo "Start time: $(date)"
echo "Exp60 type: ${EXP60_TYPE}"
echo ""

# 檢查 exp_1222 進程的函數
check_exp1222_running() {
    # 方法 1: 檢查 run_all_sequential.sh 進程
    if pgrep -f "run_all_sequential.sh" > /dev/null 2>&1; then
        return 0  # 還在運行
    fi

    # 方法 2: 檢查 exp_1222/train.py 進程
    if pgrep -f "exp_1222/train.py" > /dev/null 2>&1; then
        return 0  # 還在運行
    fi

    return 1  # 已結束
}

# 等待 exp_1222 完成
echo "Waiting for exp_1222 to complete..."
echo ""

WAIT_COUNT=0
CHECK_INTERVAL=60  # 每 60 秒檢查一次

while check_exp1222_running; do
    WAIT_COUNT=$((WAIT_COUNT + 1))
    ELAPSED_MIN=$((WAIT_COUNT * CHECK_INTERVAL / 60))
    echo "[$(date '+%H:%M:%S')] exp_1222 still running... (waited ${ELAPSED_MIN} min)"
    sleep ${CHECK_INTERVAL}
done

echo ""
echo "=============================================="
echo "exp_1222 completed!"
echo "Time: $(date)"
echo "=============================================="
echo ""

# 等待 30 秒確保資源釋放
echo "Waiting 30s for GPU memory to be released..."
sleep 30

# 運行 Exp60
echo ""
echo "=============================================="
echo "Starting Exp60 (${EXP60_TYPE})..."
echo "Time: $(date)"
echo "=============================================="
echo ""

cd /home/sbplab/ruizi/WavTokenize-self-supervised

if [ "${EXP60_TYPE}" == "crossattn" ]; then
    echo "Running: exp_1223/run_exp60_speaker_crossattn.sh"
    bash exp_1223/run_exp60_speaker_crossattn.sh
else
    echo "Running: exp_1223/run_exp60_speaker_film.sh"
    bash exp_1223/run_exp60_speaker_film.sh
fi

echo ""
echo "=============================================="
echo "Exp60 (${EXP60_TYPE}) completed!"
echo "End time: $(date)"
echo "=============================================="

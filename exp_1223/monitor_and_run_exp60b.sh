#!/bin/bash
# 監控腳本: 等待 exp60 (FiLM) 完成後自動運行 exp60b (CrossAttention with normalization)
#
# 使用方式:
#   nohup bash exp_1223/monitor_and_run_exp60b.sh > exp_1223/monitor_exp60b.log 2>&1 &
#
# 注意:
# - 此腳本監控 exp60_speaker_film 訓練進程
# - 當 epoch 200 完成後，自動啟動 CrossAttention 實驗
# - CrossAttention 會使用新的 speaker embedding L2 normalization

set -e

echo "=============================================="
echo "Monitor Script: Wait for exp60 FiLM → Run exp60b CrossAttn"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# 檢查 exp60 進程的函數
check_exp60_running() {
    # 檢查 exp60_speaker_film 進程
    if pgrep -f "exp_name exp60_speaker_film" > /dev/null 2>&1; then
        return 0  # 還在運行
    fi

    # 備用: 檢查 train_speaker.py + film
    if pgrep -f "train_speaker.py.*--speaker_condition_type film" > /dev/null 2>&1; then
        return 0  # 還在運行
    fi

    return 1  # 已結束
}

# 獲取當前 epoch
get_current_epoch() {
    if [ -f exp_1223/exp60.log ]; then
        grep "^Epoch [0-9]*/200" exp_1223/exp60.log | tail -1 | sed 's/Epoch \([0-9]*\).*/\1/' || echo "?"
    else
        echo "?"
    fi
}

# 等待 exp60 完成
echo "Waiting for exp60 (FiLM) to complete..."
echo ""

WAIT_COUNT=0
CHECK_INTERVAL=60  # 每 60 秒檢查一次

while check_exp60_running; do
    WAIT_COUNT=$((WAIT_COUNT + 1))
    ELAPSED_MIN=$((WAIT_COUNT * CHECK_INTERVAL / 60))
    CURRENT_EPOCH=$(get_current_epoch)
    echo "[$(date '+%H:%M:%S')] exp60 still running... epoch ${CURRENT_EPOCH}/200 (waited ${ELAPSED_MIN} min)"
    sleep ${CHECK_INTERVAL}
done

echo ""
echo "=============================================="
echo "exp60 (FiLM) completed!"
echo "Time: $(date)"
echo "=============================================="
echo ""

# 等待 30 秒確保資源釋放
echo "Waiting 30s for GPU memory to be released..."
sleep 30

# 運行 exp60b (CrossAttention)
echo ""
echo "=============================================="
echo "Starting exp60b (CrossAttention with speaker normalization)..."
echo "Time: $(date)"
echo "=============================================="
echo ""

cd /home/sbplab/ruizi/WavTokenize-self-supervised

# 運行 CrossAttention 實驗
echo "Running: exp_1223/run_exp60_speaker_crossattn.sh"
bash exp_1223/run_exp60_speaker_crossattn.sh

echo ""
echo "=============================================="
echo "exp60b (CrossAttention) completed!"
echo "End time: $(date)"
echo "=============================================="

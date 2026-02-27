#!/usr/bin/env bash
# watch_and_launch_0225d.sh
# 監控 exp_0225b，結束後在同一個 GPU（cuda:0）自動啟動 exp_0225d
#
# 用法：
#   bash exp_0225/watch_and_launch_0225d.sh
#   或背景執行：nohup bash exp_0225/watch_and_launch_0225d.sh > /tmp/watch_0225d.log 2>&1 &

set -euo pipefail

BASE=/home/sbplab/ruizi/WavTokenize-feature-analysis
CONDA_PYTHON=/home/sbplab/miniconda3/envs/test/bin/python
LOG_0225D=/tmp/exp0225d_train.log
WATCH_INTERVAL=60
DEVICE=cuda:0

export PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:${PYTHONPATH:-}

cd "$BASE"

echo "=============================================="
echo "[0225d watcher] 啟動監控：等待 exp_0225b 完成"
echo "[0225d watcher] 時間：$(date)"
echo "=============================================="

# -------------------------------------------------------
# 等待 exp_0225b 結束：偵測 train.log 含 "Training complete"
# 且 best_model_val_total.pt 存在
# -------------------------------------------------------
RUN_DIR_B=""
RUN_DIR_D=""
ENCODER_CKPT=""

while true; do
    # 找最新的 exp_0225b run 目錄
    LATEST_B=$(ls -dt "$BASE/exp_0225/runs/no_vq_scratch_dec_lora_epoch_"* 2>/dev/null | head -1 || true)

    if [[ -z "$LATEST_B" ]]; then
        echo "[$(date +%H:%M:%S)] 找不到 exp_0225b run 目錄，等待中..."
        sleep "$WATCH_INTERVAL"
        continue
    fi

    TRAIN_LOG_B="$LATEST_B/train.log"

    DONE=false
    if [[ -f "$TRAIN_LOG_B" ]] && grep -q "Training complete" "$TRAIN_LOG_B" 2>/dev/null; then
        DONE=true
        echo "[$(date +%H:%M:%S)] 偵測到 'Training complete'，exp_0225b 已完成！"
    fi

    if [[ "$DONE" == "true" ]]; then
        RUN_DIR_B="$LATEST_B"
        break
    fi

    # 顯示進度
    if [[ -f "$TRAIN_LOG_B" ]]; then
        LAST_LINE=$(grep -a "Epoch [0-9]*/300" "$TRAIN_LOG_B" 2>/dev/null | tail -1 || echo "(尚無 epoch 資訊)")
        echo "[$(date +%H:%M:%S)] exp_0225b 進行中... | $LAST_LINE"
    else
        echo "[$(date +%H:%M:%S)] exp_0225b train.log 尚未出現..."
    fi

    sleep "$WATCH_INTERVAL"
done

echo ""
echo "=============================================="
echo "[0225d watcher] exp_0225b 完成！"
echo "[0225d watcher] Run dir: $RUN_DIR_B"
echo "[0225d watcher] 準備啟動 exp_0225d（Frozen Disc FM Loss）..."
echo "[0225d watcher] GPU: $DEVICE"
echo "[0225d watcher] Log: $LOG_0225D"
echo "[0225d watcher] 時間：$(date)"
echo "=============================================="
echo ""

# -------------------------------------------------------
# 找 exp_0225a encoder ckpt（0225d 用同一個 encoder）
# -------------------------------------------------------
ENCODER_CKPT=$(ls -t "$BASE/exp_0225/runs/no_vq_scratch_epoch_"*/best_model_val_total.pt 2>/dev/null | head -1 || true)

if [[ -z "$ENCODER_CKPT" ]]; then
    echo "[0225d watcher] ERROR: 找不到 exp_0225a encoder ckpt，退出"
    exit 1
fi

echo "[0225d watcher] Encoder ckpt: $ENCODER_CKPT"

# -------------------------------------------------------
# 啟動 exp_0225d
# -------------------------------------------------------
nohup "$CONDA_PYTHON" \
    "$BASE/exp_0225/train_no_vq_scratch_decoder_lora_fm.py" \
    --mode epoch \
    --epochs 300 \
    --device "$DEVICE" \
    --encoder_ckpt "$ENCODER_CKPT" \
    --lambda_fm 2.0 \
    > "$LOG_0225D" 2>&1 &

PID_0225D=$!
echo "[0225d watcher] exp_0225d 已啟動，PID=$PID_0225D"
echo "[0225d watcher] 查看 log：tail -f $LOG_0225D"
echo "$PID_0225D" > /tmp/exp0225d.pid

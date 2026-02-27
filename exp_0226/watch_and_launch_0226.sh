#!/usr/bin/env bash
# watch_and_launch_0226.sh
# 監控 exp_0225c，結束後在 cuda:1 啟動 exp_0226（E2E LoRA）
#
# 用法：
#   bash exp_0226/watch_and_launch_0226.sh
#   或背景執行：nohup bash exp_0226/watch_and_launch_0226.sh > /tmp/watch_0226.log 2>&1 &

set -euo pipefail

BASE=/home/sbplab/ruizi/WavTokenize-feature-analysis
CONDA_PYTHON=/home/sbplab/miniconda3/envs/test/bin/python
LOG_0226=/tmp/exp0226_train.log
WATCH_INTERVAL=60
DEVICE=cuda:1

export PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:${PYTHONPATH:-}

cd "$BASE"

echo "=============================================="
echo "[0226 watcher] 啟動監控：等待 exp_0225c 完成"
echo "[0226 watcher] 時間：$(date)"
echo "=============================================="

# -------------------------------------------------------
# 等待 exp_0225c 結束
# -------------------------------------------------------
while true; do
    LATEST_C=$(ls -dt "$BASE/exp_0225/runs/no_vq_scratch_dec_lora_phase_epoch_"* 2>/dev/null | head -1 || true)

    if [[ -z "$LATEST_C" ]]; then
        echo "[$(date +%H:%M:%S)] 找不到 exp_0225c run 目錄，等待中..."
        sleep "$WATCH_INTERVAL"
        continue
    fi

    TRAIN_LOG_C="$LATEST_C/train.log"

    if [[ -f "$TRAIN_LOG_C" ]] && grep -q "Training complete" "$TRAIN_LOG_C" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] 偵測到 'Training complete'，exp_0225c 已完成！"
        break
    fi

    if [[ -f "$TRAIN_LOG_C" ]]; then
        LAST_EP=$(grep -a "Epoch [0-9]*/[0-9]*" "$TRAIN_LOG_C" 2>/dev/null | tail -1 || echo "(尚無 epoch 資訊)")
        echo "[$(date +%H:%M:%S)] exp_0225c 進行中... | $LAST_EP"
    else
        echo "[$(date +%H:%M:%S)] exp_0225c train.log 尚未出現..."
    fi

    sleep "$WATCH_INTERVAL"
done

echo ""
echo "=============================================="
echo "[0226 watcher] exp_0225c 完成！"
echo "[0226 watcher] 啟動 exp_0226（E2E LoRA，cuda:1）..."
echo "[0226 watcher] Log: $LOG_0226"
echo "[0226 watcher] 時間：$(date)"
echo "=============================================="
echo ""

# exp_0225a encoder ckpt
ENCODER_CKPT="$BASE/exp_0225/runs/no_vq_scratch_epoch_20260224_032104/best_model_val_total.pt"

if [[ ! -f "$ENCODER_CKPT" ]]; then
    echo "[0226 watcher] ERROR: 找不到 exp_0225a encoder ckpt: $ENCODER_CKPT"
    exit 1
fi

echo "[0226 watcher] Encoder ckpt: $ENCODER_CKPT"

nohup "$CONDA_PYTHON" \
    "$BASE/exp_0226/train_no_vq_e2e.py" \
    --mode epoch \
    --epochs 300 \
    --device "$DEVICE" \
    --encoder_ckpt "$ENCODER_CKPT" \
    > "$LOG_0226" 2>&1 &

PID_0226=$!
echo "[0226 watcher] exp_0226 已啟動，PID=$PID_0226"
echo "[0226 watcher] 查看 log：tail -f $LOG_0226"
echo "$PID_0226" > /tmp/exp0226.pid

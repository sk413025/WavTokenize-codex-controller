#!/usr/bin/env bash
# watch_and_launch_0226a.sh
# 監控 exp_0226 E2E，結束後啟動 exp_0226a（Encoder + Feature Alignment）
#
# 用法：
#   nohup bash families/deps/feat_align/watch_and_launch_0226a.sh > /tmp/watch_0226a.log 2>&1 &
#
# 注意：啟動腳本已移到 quarantine/python/，此 watcher 僅保留歷史追溯用途。

set -euo pipefail

BASE=/home/sbplab/ruizi/WavTokenize-feature-analysis
CONDA_PYTHON=/home/sbplab/miniconda3/envs/test/bin/python
LOG_0226A=/tmp/exp0226a_train.log
WATCH_INTERVAL=60

export PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:${PYTHONPATH:-}

cd "$BASE"

echo "=============================================="
echo "[0226a watcher] 啟動監控：等待 exp_0226 E2E 完成"
echo "[0226a watcher] 時間：$(date)"
echo "=============================================="

# -------------------------------------------------------
# 等待 exp_0226 E2E 結束
# -------------------------------------------------------
while true; do
    LATEST=$(ls -dt "$BASE/families/deps/feat_align/runs/no_vq_e2e_epoch_"* 2>/dev/null | head -1 || true)

    if [[ -z "$LATEST" ]]; then
        echo "[$(date +%H:%M:%S)] 找不到 exp_0226 run 目錄，等待中..."
        sleep "$WATCH_INTERVAL"
        continue
    fi

    TRAIN_LOG="$LATEST/train.log"

    if [[ -f "$TRAIN_LOG" ]] && grep -q "Training complete" "$TRAIN_LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] 偵測到 'Training complete'，exp_0226 E2E 已完成！"
        break
    fi

    if [[ -f "$TRAIN_LOG" ]]; then
        LAST_EP=$(grep -a "^Epoch [0-9]*/300" "$TRAIN_LOG" 2>/dev/null | tail -1 || echo "(尚無 epoch 資訊)")
        echo "[$(date +%H:%M:%S)] exp_0226 E2E 進行中... | $LAST_EP"
    else
        echo "[$(date +%H:%M:%S)] exp_0226 train.log 尚未出現..."
    fi

    sleep "$WATCH_INTERVAL"
done

# -------------------------------------------------------
# 偵測空閒 GPU
# -------------------------------------------------------
# exp_0226 E2E 跑在 cuda:1，結束後 cuda:1 應該空出來
DEVICE=cuda:1

echo ""
echo "=============================================="
echo "[0226a watcher] exp_0226 E2E 完成！"
echo "[0226a watcher] 啟動 exp_0226a（Encoder + Feature Alignment，$DEVICE）..."
echo "[0226a watcher] Log: $LOG_0226A"
echo "[0226a watcher] 時間：$(date)"
echo "=============================================="
echo ""

# encoder ckpt：exp_0225a best_model_val_total.pt
ENCODER_CKPT="$BASE/families/deps/no_vq_scratch/runs/no_vq_scratch_epoch_20260224_032104/best_model_val_total.pt"

if [[ ! -f "$ENCODER_CKPT" ]]; then
    echo "[0226a watcher] ERROR: 找不到 exp_0225a encoder ckpt: $ENCODER_CKPT"
    exit 1
fi

echo "[0226a watcher] Encoder ckpt: $ENCODER_CKPT"

nohup "$CONDA_PYTHON" \
    "$BASE/quarantine/python/families/deps/feat_align/train_enc_feat_align.py" \
    --mode epoch \
    --epochs 300 \
    --device "$DEVICE" \
    --encoder_ckpt "$ENCODER_CKPT" \
    --lambda_feat 1.0 \
    > "$LOG_0226A" 2>&1 &

PID_0226A=$!
echo "[0226a watcher] exp_0226a 已啟動，PID=$PID_0226A"
echo "[0226a watcher] 查看 log：tail -f $LOG_0226A"
echo "$PID_0226A" > /tmp/exp0226a.pid

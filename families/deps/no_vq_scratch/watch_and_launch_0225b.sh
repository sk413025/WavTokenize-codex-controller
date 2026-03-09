#!/usr/bin/env bash
# watch_and_launch_0225b.sh
# 監控 exp_0225a（quarantine/python/.../train_no_vq_scratch.py），結束後自動啟動 exp_0225b（Decoder LoRA）
#
# 用法：bash families/deps/no_vq_scratch/watch_and_launch_0225b.sh
# 或背景執行：nohup bash families/deps/no_vq_scratch/watch_and_launch_0225b.sh > /tmp/watch_0225b.log 2>&1 &

set -euo pipefail

BASE=/home/sbplab/ruizi/WavTokenize-feature-analysis
CONDA_PYTHON=/home/sbplab/miniconda3/envs/test/bin/python
LOG_0225B=/tmp/exp0225b_train.log
WATCH_INTERVAL=60   # 每 60 秒檢查一次
DEVICE=cuda:0

export PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:${PYTHONPATH:-}

cd "$BASE"

echo "=============================================="
echo "[0225 watcher] 啟動監控：等待 exp_0225a 完成"
echo "[0225 watcher] 時間：$(date)"
echo "=============================================="

# -------------------------------------------------------
# 等待 exp_0225a 結束：偵測 best_model_val_total.pt 出現
# 且 train.log 中含 "Training complete" 或 PID 不存在
# -------------------------------------------------------
RUN_DIR=""
CKPT_PATH=""

while true; do
    # 找最新的 exp_0225a run 目錄（只找正式訓練，排除 smoke）
    LATEST_RUN=$(ls -dt "$BASE/families/deps/no_vq_scratch/runs/no_vq_scratch_epoch_"* 2>/dev/null | head -1 || true)

    if [[ -z "$LATEST_RUN" ]]; then
        echo "[$(date +%H:%M:%S)] 找不到 exp_0225a run 目錄，等待中..."
        sleep "$WATCH_INTERVAL"
        continue
    fi

    CKPT_CANDIDATE="$LATEST_RUN/best_model_val_total.pt"
    TRAIN_LOG="$LATEST_RUN/train.log"

    # 檢查訓練是否完成
    DONE=false
    if [[ -f "$TRAIN_LOG" ]] && grep -q "Training complete\|Epoch 300\|epoch 300" "$TRAIN_LOG" 2>/dev/null; then
        DONE=true
        echo "[$(date +%H:%M:%S)] 偵測到 'Training complete'，exp_0225a 已完成！"
    fi

    if [[ "$DONE" == "true" ]] && [[ -f "$CKPT_CANDIDATE" ]]; then
        RUN_DIR="$LATEST_RUN"
        CKPT_PATH="$CKPT_CANDIDATE"
        break
    fi

    # 顯示進度
    if [[ -f "$TRAIN_LOG" ]]; then
        LAST_LINE=$(tail -1 "$TRAIN_LOG" 2>/dev/null || echo "(空)")
        echo "[$(date +%H:%M:%S)] exp_0225a 進行中... | $LAST_LINE"
    else
        echo "[$(date +%H:%M:%S)] exp_0225a 訓練 log 尚未出現..."
    fi

    sleep "$WATCH_INTERVAL"
done

echo ""
echo "=============================================="
echo "[0225 watcher] exp_0225a 完成！"
echo "[0225 watcher] Run dir:  $RUN_DIR"
echo "[0225 watcher] Ckpt:     $CKPT_PATH"
echo "[0225 watcher] 啟動 exp_0225b（Decoder LoRA）..."
echo "[0225 watcher] Log:      $LOG_0225B"
echo "[0225 watcher] 時間：$(date)"
echo "=============================================="
echo ""

# -------------------------------------------------------
# 啟動 exp_0225b
# -------------------------------------------------------
nohup "$CONDA_PYTHON" \
    "$BASE/quarantine/python/families/deps/no_vq_scratch/train_no_vq_scratch_decoder_lora.py" \
    --mode epoch \
    --epochs 300 \
    --device "$DEVICE" \
    --encoder_ckpt "$CKPT_PATH" \
    > "$LOG_0225B" 2>&1 &

PID_0225B=$!
echo "[0225 watcher] exp_0225b 已啟動，PID=$PID_0225B"
echo "[0225 watcher] 查看 log：tail -f $LOG_0225B"
echo "$PID_0225B" > /tmp/exp0225b.pid

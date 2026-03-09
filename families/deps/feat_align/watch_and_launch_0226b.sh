#!/usr/bin/env bash
# watch_and_launch_0226b.sh
# 監控 exp_0226a（Encoder + Feature Alignment）完成後，啟動 exp_0226b（+ HF Mel Loss）
#
# 用法：
#   nohup bash families/deps/feat_align/watch_and_launch_0226b.sh > /tmp/watch_0226b.log 2>&1 &

set -euo pipefail

BASE=/home/sbplab/ruizi/WavTokenize-feature-analysis
CONDA_PYTHON=/home/sbplab/miniconda3/envs/test/bin/python
LOG_0226B=/tmp/exp0226b_train.log
WATCH_INTERVAL=60

export PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:${PYTHONPATH:-}

cd "$BASE"

echo "=============================================="
echo "[0226b watcher] 啟動監控：等待 exp_0226a 完成"
echo "[0226b watcher] 時間：$(date)"
echo "=============================================="

# -------------------------------------------------------
# 等待 exp_0226a 結束（偵測 "Training complete" 字樣）
# -------------------------------------------------------
while true; do
    LATEST=$(ls -dt "$BASE/families/deps/feat_align/runs/enc_feat_align_epoch_"* 2>/dev/null | head -1 || true)

    if [[ -z "$LATEST" ]]; then
        echo "[$(date +%H:%M:%S)] 找不到 exp_0226a run 目錄，等待中..."
        sleep "$WATCH_INTERVAL"
        continue
    fi

    TRAIN_LOG="$LATEST/train.log"

    if [[ -f "$TRAIN_LOG" ]] && grep -q "Training complete" "$TRAIN_LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] 偵測到 'Training complete'，exp_0226a 已完成！"
        echo "[$(date +%H:%M:%S)] Run dir: $LATEST"
        break
    fi

    if [[ -f "$TRAIN_LOG" ]]; then
        LAST_EP=$(grep -a "^Epoch [0-9]*/300" "$TRAIN_LOG" 2>/dev/null | tail -1 || echo "(尚無 epoch 資訊)")
        echo "[$(date +%H:%M:%S)] exp_0226a 進行中... | $LAST_EP"
    else
        echo "[$(date +%H:%M:%S)] exp_0226a train.log 尚未出現..."
    fi

    sleep "$WATCH_INTERVAL"
done

# -------------------------------------------------------
# 確認 GPU 狀態（0226a 跑在 cuda:1，結束後空出）
# -------------------------------------------------------
DEVICE=cuda:1

echo ""
echo "=============================================="
echo "[0226b watcher] exp_0226a 完成！"
echo "[0226b watcher] 啟動 exp_0226b（Encoder + FeatAlign + HF Mel，$DEVICE）..."
echo "[0226b watcher] Log: $LOG_0226B"
echo "[0226b watcher] 時間：$(date)"
echo "=============================================="
echo ""

# encoder ckpt：exp_0225a best_model_val_total.pt（與 0226a 相同起點）
# 注意：不用 0226a 的 best_model，是為了讓消融比較更 clean（相同 encoder 起點）
# 若想繼續 0226a 的 best，改為下方 ENCODER_CKPT_0226A
ENCODER_CKPT="$BASE/families/deps/no_vq_scratch/runs/no_vq_scratch_epoch_20260224_032104/best_model_val_total.pt"

# （可選）從 0226a best_model_val_total.pt 繼續訓練（不同消融語意）
# ENCODER_CKPT=$(ls -t "$BASE"/families/deps/feat_align/runs/enc_feat_align_epoch_*/best_model_val_total.pt 2>/dev/null | head -1 || true)
# if [[ -z "$ENCODER_CKPT" ]]; then
#     echo "[0226b watcher] ERROR: 找不到 0226a best_model_val_total.pt"
#     exit 1
# fi

if [[ ! -f "$ENCODER_CKPT" ]]; then
    echo "[0226b watcher] ERROR: 找不到 encoder ckpt: $ENCODER_CKPT"
    exit 1
fi

echo "[0226b watcher] Encoder ckpt: $ENCODER_CKPT"

nohup "$CONDA_PYTHON" \
    "$BASE/families/deps/feat_align/train_enc_hf_mel.py" \
    --mode epoch \
    --epochs 300 \
    --device "$DEVICE" \
    --encoder_ckpt "$ENCODER_CKPT" \
    --lambda_wav 1.0 \
    --lambda_stft 1.0 \
    --lambda_mel 45.0 \
    --lambda_feat 1.0 \
    --lambda_hf 45.0 \
    --hf_bin_start 40 \
    > "$LOG_0226B" 2>&1 &

PID_0226B=$!
echo "[0226b watcher] exp_0226b 已啟動，PID=$PID_0226B"
echo "[0226b watcher] 查看 log：tail -f $LOG_0226B"
echo "$PID_0226B" > /tmp/exp0226b.pid

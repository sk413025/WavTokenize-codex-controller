#!/usr/bin/env bash
# watch_and_launch_0227.sh
# 監控 exp_0225d（DecLoRA + FM）完成後，啟動 exp_0227（EncOnly + Frozen MRD FM）
# exp_0225d 跑在 cuda:0，結束後 cuda:0 空出來給 exp_0227
#
# 用法：
#   nohup bash exp_0227/watch_and_launch_0227.sh > /tmp/watch_0227.log 2>&1 &

set -euo pipefail

BASE=/home/sbplab/ruizi/WavTokenize-feature-analysis
CONDA_PYTHON=/home/sbplab/miniconda3/envs/test/bin/python
LOG_0227=/tmp/exp0227_train.log
WATCH_INTERVAL=60

export PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:${PYTHONPATH:-}

cd "$BASE"

echo "=============================================="
echo "[0227 watcher] 啟動監控：等待 exp_0225d 完成"
echo "[0227 watcher] 時間：$(date)"
echo "=============================================="

# -------------------------------------------------------
# 等待 exp_0225d 結束（偵測 "Training complete"）
# -------------------------------------------------------
EXP0225D_LOG="$BASE/exp_0225/runs/no_vq_scratch_dec_lora_fm_epoch_20260225_095842/train.log"

while true; do
    if [[ ! -f "$EXP0225D_LOG" ]]; then
        echo "[$(date +%H:%M:%S)] 找不到 exp_0225d train.log，等待中..."
        sleep "$WATCH_INTERVAL"
        continue
    fi

    if grep -q "Training complete" "$EXP0225D_LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] 偵測到 'Training complete'，exp_0225d 已完成！"
        break
    fi

    LAST_EP=$(grep "^Epoch [0-9]*/300" "$EXP0225D_LOG" 2>/dev/null | tail -n 1 || echo "(尚無 epoch 資訊)")
    echo "[$(date +%H:%M:%S)] exp_0225d 進行中... | $LAST_EP"

    sleep "$WATCH_INTERVAL"
done

# -------------------------------------------------------
# 啟動 exp_0227
# -------------------------------------------------------
DEVICE=cuda:1

echo ""
echo "=============================================="
echo "[0227 watcher] exp_0226b 完成！"
echo "[0227 watcher] 啟動 exp_0227（EncOnly + FeatAlign + Frozen MRD FM，$DEVICE）..."
echo "[0227 watcher] Log: $LOG_0227"
echo "[0227 watcher] 時間：$(date)"
echo "=============================================="
echo ""

# encoder ckpt：exp_0225a best_model_val_total.pt（統一起點，消融比較清楚）
ENCODER_CKPT="$BASE/exp_0225/runs/no_vq_scratch_epoch_20260224_032104/best_model_val_total.pt"

if [[ ! -f "$ENCODER_CKPT" ]]; then
    echo "[0227 watcher] ERROR: 找不到 encoder ckpt: $ENCODER_CKPT"
    exit 1
fi

echo "[0227 watcher] Encoder ckpt: $ENCODER_CKPT"

nohup "$CONDA_PYTHON" \
    "$BASE/exp_0227/train_enc_mrd_fm.py" \
    --mode epoch \
    --epochs 300 \
    --device "$DEVICE" \
    --encoder_ckpt "$ENCODER_CKPT" \
    --lambda_wav  1.0 \
    --lambda_stft 1.0 \
    --lambda_mel  45.0 \
    --lambda_feat 1.0 \
    --lambda_fm   2.0 \
    > "$LOG_0227" 2>&1 &

PID_0227=$!
echo "[0227 watcher] exp_0227 已啟動，PID=$PID_0227"
echo "[0227 watcher] 查看 log：tail -f $LOG_0227"
echo "$PID_0227" > /tmp/exp0227.pid

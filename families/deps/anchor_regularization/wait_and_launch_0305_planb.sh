#!/usr/bin/env bash
# 監控 exp_0305b expA 結束，完成後自動在 cuda:0 啟動 exp_0305 plan_b
# 用法: bash families/deps/anchor_regularization/wait_and_launch_0305_planb.sh &
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_A="${ROOT_DIR}/families/deps/anchor_regularization/runs/expA_tail_lock_origWav_20260305_002937/train.log"
CHECK_INTERVAL=120  # 每 2 分鐘檢查一次

echo "======================================================"
echo "[$(date)] 監控 expA train.log: ${LOG_A}"
echo "expA 結束後自動啟動 exp_0305 plan_b (cuda:0)"
echo "======================================================"

# 等待 expA 完成（train.log 出現 "Training complete" 或 Epoch 120/120）
while true; do
    if grep -qE "^Epoch 120/120|Training complete|KeyboardInterrupt" "${LOG_A}" 2>/dev/null; then
        echo "[$(date)] expA 已完成，準備啟動 exp_0305 plan_b ..."
        break
    fi
    LATEST=$(grep -E "^Epoch [0-9]+/120" "${LOG_A}" 2>/dev/null | tail -1 || echo "尚無完整 epoch")
    echo "[$(date)] expA 進度: ${LATEST}"
    sleep ${CHECK_INTERVAL}
done

# 額外等 30 秒讓 expA 完整釋放 GPU 記憶體
sleep 30

# 確認 cuda:0 已釋放 (used < 1000 MiB)
GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "[$(date)] cuda:0 目前 GPU memory used: ${GPU0_MEM} MiB"
if [ "${GPU0_MEM}" -gt 1000 ]; then
    echo "[$(date)] 警告: cuda:0 記憶體仍有 ${GPU0_MEM} MiB，等待 60 秒..."
    sleep 60
fi

# 啟動 exp_0305 plan_b
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PLANB="${ROOT_DIR}/families/deps/selective_lora/nohup_planb_${TS}.log"
echo "[$(date)] 啟動 exp_0305 plan_b => ${LOG_PLANB}"

cd "${ROOT_DIR}"
nohup python families/deps/selective_lora/train_selective_lora.py \
    --plan plan_b \
    --mode epoch \
    --epochs 300 \
    --device cuda:0 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup 5 \
    --num_workers 2 \
    > "${LOG_PLANB}" 2>&1 &

PLANB_PID=$!
echo "[$(date)] exp_0305 plan_b PID=${PLANB_PID}，log: ${LOG_PLANB}"
echo "${PLANB_PID}" > "${ROOT_DIR}/families/deps/selective_lora/planb_pid.txt"
echo "[$(date)] 完成！可用以下指令監控："
echo "  tail -f ${LOG_PLANB}"

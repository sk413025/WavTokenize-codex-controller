#!/usr/bin/env bash
# 監控 exp_0305 plan_b 結束後，自動在空閒 GPU 啟動 plan_d
# 用法: bash families/deps/selective_lora/wait_and_launch_pland.sh &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLANB_LOG_DIR="${ROOT_DIR}/families/deps/selective_lora/runs/plan_b_epoch_20260306_052339"
CHECK_INTERVAL=120  # 每 2 分鐘檢查一次

echo "======================================================"
echo "[$(date)] 監控 plan_b: ${PLANB_LOG_DIR}/train.log"
echo "結束後自動啟動 exp_0305 plan_d（空閒 GPU）"
echo "======================================================"

while true; do
    if grep -qE "^Training complete|訓練完成|^Epoch 300/300" "${PLANB_LOG_DIR}/train.log" 2>/dev/null; then
        echo "[$(date)] plan_b 完成，準備啟動 plan_d..."
        break
    fi
    LATEST=$(grep -E "^\[Epoch +[0-9]+\]" "${PLANB_LOG_DIR}/train.log" 2>/dev/null | tail -1 || echo "尚無完整 epoch")
    echo "[$(date)] plan_b 進度: ${LATEST}"
    sleep ${CHECK_INTERVAL}
done

sleep 30  # 等 GPU 記憶體釋放

# 找空閒 GPU（used < 1000 MiB）
FREE_GPU=""
for i in 0 1; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${i})
    echo "[$(date)] GPU${i} memory used: ${MEM} MiB"
    if [ "${MEM}" -lt 1000 ]; then
        FREE_GPU=${i}
        break
    fi
done

if [ -z "${FREE_GPU}" ]; then
    echo "[$(date)] 警告：沒有空閒 GPU，等待 60 秒再試..."
    sleep 60
    for i in 0 1; do
        MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${i})
        if [ "${MEM}" -lt 1000 ]; then
            FREE_GPU=${i}
            break
        fi
    done
fi

if [ -z "${FREE_GPU}" ]; then
    echo "[$(date)] 錯誤：仍無空閒 GPU，請手動啟動"
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PLAND="${ROOT_DIR}/families/deps/selective_lora/nohup_pland_${TS}.log"
echo "[$(date)] 使用 cuda:${FREE_GPU} 啟動 plan_d => ${LOG_PLAND}"

cd "${ROOT_DIR}"
nohup /home/sbplab/miniconda3/envs/test/bin/python families/deps/selective_lora/train_selective_lora.py \
    --plan plan_d \
    --mode epoch \
    --epochs 300 \
    --device cuda:${FREE_GPU} \
    --batch_size 8 \
    --grad_accum 2 \
    --num_workers 2 \
    > "${LOG_PLAND}" 2>&1 &

PID=$!
echo "[$(date)] plan_d PID=${PID}，log: ${LOG_PLAND}"
echo "${PID}" > "${ROOT_DIR}/families/deps/selective_lora/pland_pid.txt"
echo "[$(date)] 完成！監控："
echo "  tail -f ${LOG_PLAND}"

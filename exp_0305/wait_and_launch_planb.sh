#!/usr/bin/env bash
# 監控 exp_0305b expA 結束後，自動在空閒 GPU 啟動 exp_0305 plan_b
# 用法: bash exp_0305/wait_and_launch_planb.sh &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_A="${ROOT_DIR}/exp_0305b/runs/exp0305b_tail_lock_20260305_033323/train.log"
CHECK_INTERVAL=120  # 每 2 分鐘檢查一次

echo "======================================================"
echo "[$(date)] 監控 expA: ${LOG_A}"
echo "結束後自動啟動 exp_0305 plan_b（空閒 GPU）"
echo "======================================================"

while true; do
    if grep -qE "^Training complete|^Epoch 120/120" "${LOG_A}" 2>/dev/null; then
        echo "[$(date)] expA 完成，準備啟動 exp_0305 plan_b..."
        break
    fi
    LATEST=$(grep -E "^Epoch [0-9]+/120" "${LOG_A}" 2>/dev/null | tail -1 || echo "尚無完整 epoch")
    echo "[$(date)] expA 進度: ${LATEST}"
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
LOG_PLANB="${ROOT_DIR}/exp_0305/nohup_planb_${TS}.log"
echo "[$(date)] 使用 cuda:${FREE_GPU} 啟動 exp_0305 plan_b => ${LOG_PLANB}"

cd "${ROOT_DIR}"
nohup /home/sbplab/miniconda3/envs/test/bin/python exp_0305/train_selective_lora.py \
    --plan plan_b \
    --mode epoch \
    --epochs 300 \
    --device cuda:${FREE_GPU} \
    --batch_size 8 \
    --grad_accum 2 \
    --num_workers 2 \
    > "${LOG_PLANB}" 2>&1 &

PID=$!
echo "[$(date)] plan_b PID=${PID}，log: ${LOG_PLANB}"
echo "${PID}" > "${ROOT_DIR}/exp_0305/planb_pid.txt"
echo "[$(date)] 完成！監控："
echo "  tail -f ${LOG_PLANB}"

#!/usr/bin/env bash
# 監控 exp_0305c 結束後，自動在 cuda:1 啟動 exp_0304 材質泛化正式訓練
# 用法: bash families/official/anchor_then_material/wait_and_launch_exp0304.sh &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPC_LOG="${ROOT_DIR}/families/official/anchor_then_material/nohup_expc_20260306_090111.log"
CHECK_INTERVAL=120  # 每 2 分鐘檢查一次

echo "======================================================"
echo "[$(date)] 監控 exp_0305c: ${EXPC_LOG}"
echo "結束後自動在 cuda:1 啟動 exp_0304 材質泛化正式訓練"
echo "======================================================"

while true; do
    if grep -qE "^Training complete" "${EXPC_LOG}" 2>/dev/null; then
        echo "[$(date)] exp_0305c 完成，準備啟動 families.official.material_generalization..."
        break
    fi
    LATEST=$(grep -E "^Epoch [0-9]+/300" "${EXPC_LOG}" 2>/dev/null | tail -1 || echo "尚無完整 epoch")
    echo "[$(date)] exp_0305c 進度: ${LATEST}"
    sleep ${CHECK_INTERVAL}
done

sleep 30  # 等 GPU 記憶體釋放

# 確認 cuda:1 已釋放 (used < 1000 MiB)
GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
echo "[$(date)] cuda:1 目前 GPU memory used: ${GPU1_MEM} MiB"
if [ "${GPU1_MEM}" -gt 1000 ]; then
    echo "[$(date)] 警告: cuda:1 記憶體仍有 ${GPU1_MEM} MiB，等待 60 秒..."
    sleep 60
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_EXP0304="${ROOT_DIR}/families/official/material_generalization/nohup_material_gen_${TS}.log"
echo "[$(date)] 啟動 exp_0304 材質泛化 (cuda:1) => ${LOG_EXP0304}"

cd "${ROOT_DIR}"
PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \
nohup /home/sbplab/miniconda3/envs/test/bin/python families/official/material_generalization/train_material_gen.py \
    --mode epoch \
    --epochs 300 \
    --device cuda:1 \
    --batch_size 8 \
    --grad_accum 2 \
    --num_workers 2 \
    > "${LOG_EXP0304}" 2>&1 &

PID=$!
echo "[$(date)] exp_0304 material_gen PID=${PID}，log: ${LOG_EXP0304}"
echo "${PID}" > "${ROOT_DIR}/families/official/material_generalization/material_gen_pid.txt"
echo "[$(date)] 完成！監控："
echo "  tail -f ${LOG_EXP0304}"

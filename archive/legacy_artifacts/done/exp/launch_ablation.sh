#!/usr/bin/env bash
set -euo pipefail

# 用法: bash done/exp/launch_ablation.sh <GPU_ID> <mode> [OUT_BASENAME] [BATCH]
#   <mode>: ma | dir
#   OUT_BASENAME: 預設 ablations_<mode>_100ep

GPU_ID="${1:-}"
MODE="${2:-}"
OUT_BASE="${3:-ablations_${MODE}_100ep}"
BATCH="${4:-32}"

if [[ -z "${GPU_ID}" || -z "${MODE}" ]]; then
  echo "用法: $0 <GPU_ID> <mode[ma|dir]> [OUT_BASENAME]" >&2
  exit 1
fi

cd /home/sbplab/ruizi/WavTokenize-self-supervised

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/${OUT_BASE}_${STAMP}"
mkdir -p "${OUT_DIR}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  --cache_dir /home/sbplab/ruizi/c_code/done/exp/data
  --num_epochs 100
  --batch_size "${BATCH}"
  --learning_rate 1e-4
  --speaker_tokens 4
  --output_dir "${OUT_DIR}"
)

if [[ "${MODE}" == "ma" ]]; then
  # Margin-aware 門控：低=0；中×1.8；高×0.5
  python -u done/exp/train_crossattn_gated_cached.py \
    "${COMMON_ARGS[@]}" \
    --margin_aware --low_thr 0.02 --mid_thr 0.4 --mid_amp 1.8 --high_amp 0.5 \
    |& tee "${OUT_DIR}/console.log"
elif [[ "${MODE}" == "dir" ]]; then
  # 方向性輔助 loss：中 margin 限定，權重 0.2
  python -u done/exp/train_crossattn_gated_cached.py \
    "${COMMON_ARGS[@]}" \
    --dir_loss_weight 0.2 --dir_mid_only \
    |& tee "${OUT_DIR}/console.log"
else
  echo "未知的 mode: ${MODE} (應為 ma 或 dir)" >&2
  exit 2
fi

echo "✓ 任務完成: ${OUT_DIR}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT_DIR}/families/deps/anchor_regularization/runs/dual_anchor_${TS}"
mkdir -p "${RUN_ROOT}"

# Tunables (override by env vars)
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${MODE:-epoch}"          # epoch | smoke
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"

# Anchor strengths
# A: constrain tail (L16/L17), B: constrain front+tail (L0/L1/L16/L17)
TAIL_LAMBDA="${TAIL_LAMBDA:-1.0}"   # Experiment A
FRONT_LAMBDA="${FRONT_LAMBDA:-3.0}" # Experiment B
NO_AMP="${NO_AMP:-0}"               # 1 => pass --no_amp
COMPARE_EVERY="${COMPARE_EVERY:-50}" # comparison milestone interval (epochs)

echo "=============================================================="
echo "exp_0305b dual-anchor run"
echo "RUN_ROOT=${RUN_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "DEVICE=${DEVICE} MODE=${MODE} EPOCHS=${EPOCHS}"
echo "TAIL_LAMBDA=${TAIL_LAMBDA} (L16,L17)"
echo "FRONT_LAMBDA=${FRONT_LAMBDA} (L0,L1,L16,L17)"
echo "COMPARE_EVERY=${COMPARE_EVERY}"
echo "=============================================================="

COMMON_ARGS=(
  --mode "${MODE}"
  --epochs "${EPOCHS}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum "${GRAD_ACCUM}"
  --learning_rate "${LR}"
  --min_lr "${MIN_LR}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --num_workers "${NUM_WORKERS}"
)
if [[ "${NO_AMP}" == "1" ]]; then
  COMMON_ARGS+=(--no_amp)
fi

echo "[1/2] tail_lock (L16/L17)"
"${PYTHON_BIN}" families/deps/anchor_regularization/train_0224a_anchor.py \
  "${COMMON_ARGS[@]}" \
  --preset tail_lock \
  --lambda_anchor "${TAIL_LAMBDA}" \
  --output_dir "${RUN_ROOT}/tail_lock_L16L17"

echo "[2/2] front_tail_lock (L0/L1/L16/L17)"
"${PYTHON_BIN}" families/deps/anchor_regularization/train_0224a_anchor.py \
  "${COMMON_ARGS[@]}" \
  --preset front_tail_lock \
  --lambda_anchor "${FRONT_LAMBDA}" \
  --output_dir "${RUN_ROOT}/front_tail_lock_L0L1L16L17"

echo "[3/3] compare"
"${PYTHON_BIN}" families/deps/anchor_regularization/compare_exp0305b_dual.py \
  --tail_dir "${RUN_ROOT}/tail_lock_L16L17" \
  --front_dir "${RUN_ROOT}/front_tail_lock_L0L1L16L17" \
  --output_dir "${RUN_ROOT}/comparison" \
  --milestone_interval "${COMPARE_EVERY}"

echo "Done. Results in: ${RUN_ROOT}"

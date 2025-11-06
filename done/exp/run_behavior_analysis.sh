#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
RESULTS_DIR="${2:-}"
EPOCHS_STR="${3:-}"
K_TOPK="${4:-5}"
BATCH="${5:-16}"
CACHE_DIR="${6:-/home/sbplab/ruizi/c_code/done/exp/data}"

if [[ -z "${RESULTS_DIR}" ]]; then
  echo "Usage: $0 <gpu_id> <results_dir> <\"epochs list\"> [k_topk] [batch] [cache_dir]" >&2
  exit 1
fi

if [[ -z "${EPOCHS_STR}" ]]; then
  echo "EPOCHS_STR not provided, defaulting to \"10 20 30 40 50\"" >&2
  EPOCHS_STR="10 20 30 40 50"
fi

LOG_DIR="${RESULTS_DIR}/analysis"
mkdir -p "${LOG_DIR}"
JOB_LOG="${LOG_DIR}/analysis_gpu${GPU_ID}_job.log"

echo "[run_behavior_analysis] results_dir=${RESULTS_DIR} epochs=[${EPOCHS_STR}] k=${K_TOPK} batch=${BATCH} cache=${CACHE_DIR}" | tee -a "${JOB_LOG}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python -u done/exp/analyze_influence_breakdown.py \
  --results_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --epochs ${EPOCHS_STR} \
  --batch_size "${BATCH}" 2>&1 | tee -a "${JOB_LOG}"

python -u done/exp/analyze_margins_topk.py \
  --results_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --epochs ${EPOCHS_STR} \
  --k "${K_TOPK}" \
  --batch_size "${BATCH}" 2>&1 | tee -a "${JOB_LOG}"

python -u done/exp/analyze_logit_shift_geometry.py \
  --results_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --epochs ${EPOCHS_STR} \
  --batch_size "${BATCH}" 2>&1 | tee -a "${JOB_LOG}"

# Attention entropy (for models exposing attention)
python -u done/exp/analyze_attention_entropy.py \
  --results_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --epochs ${EPOCHS_STR} \
  --batch_size "${BATCH}" 2>&1 | tee -a "${JOB_LOG}"

# Gate distribution (gated models only; script auto-skips otherwise)
python -u done/exp/analyze_gate_distribution.py \
  --results_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --epochs ${EPOCHS_STR} \
  --batch_size "${BATCH}" 2>&1 | tee -a "${JOB_LOG}"

echo "[run_behavior_analysis] ✓ Completed for ${RESULTS_DIR}" | tee -a "${JOB_LOG}"

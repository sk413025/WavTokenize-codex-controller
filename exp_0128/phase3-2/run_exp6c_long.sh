#!/bin/bash

# ============================================================
# exp_0128 Phase 3-2: Exp 6c (LONG) - EMA + dead-code reset
# ============================================================

set -e
set -o pipefail

if [ -z "${1}" ]; then
  echo "Usage: bash exp_0128/phase3-2/run_exp6c_long.sh <dead_code_threshold> [beta_commit]"
  echo "Example: bash exp_0128/phase3-2/run_exp6c_long.sh 2 1.0"
  exit 2
fi

DEAD_CODE_THRESHOLD="${1}"
BETA_COMMIT="${2:-1.0}"

TH_TAG="${DEAD_CODE_THRESHOLD}"
BETA_TAG="${BETA_COMMIT//./p}"

source ~/miniconda3/etc/profile.d/conda.sh
set +e
conda activate test
set -e

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# Prefer 2080 Ti (physical GPU 1). Use cuda:0 inside the process.
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="exp_0128/phase3-2/run_exp6c_long_ema_th${TH_TAG}_beta${BETA_TAG}_${TIMESTAMP}"

echo "============================================================"
echo "Starting Exp 6c (LONG): EMA + dead-code reset (Phase 3-2)"
echo "============================================================"
echo "dead_code_threshold=${DEAD_CODE_THRESHOLD}"
echo "beta_commit=${BETA_COMMIT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (process sees cuda:0)"
echo "Output: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "============================================================"

PYTHONUNBUFFERED=1 python exp_0128/phase3/residual_vq/train_rvq_short_run.py \
  --steps 1000 \
  --batch_size 8 \
  --grad_accum 2 \
  --lr 1e-4 \
  --n_rvq_layers 4 \
  --rvq_codebook_size 1024 \
  --lambda_quant 1.0 \
  --lambda_pre 0.0 \
  --lambda_inter 0.5 \
  --beta_commit ${BETA_COMMIT} \
  --lambda_codebook 0.0 \
  --rvq_update ema \
  --ema_decay 0.99 \
  --ema_eps 1e-5 \
  --ema_dead_code_threshold ${DEAD_CODE_THRESHOLD} \
  --inter_warmup_steps 0 \
  --early_stop_on_collapse \
  --output_dir ${OUTPUT_DIR} \
  --seed 42 \
  --device cuda:0 \
  --eval_interval 200 \
  |& tee ${OUTPUT_DIR}.log

echo ""
echo "============================================================"
echo "Exp 6c (LONG) (th=${DEAD_CODE_THRESHOLD}, beta_commit=${BETA_COMMIT}) Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log saved to: ${OUTPUT_DIR}.log"
echo "============================================================"


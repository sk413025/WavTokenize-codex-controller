#!/bin/bash

# ============================================================
# exp_0128 Phase 3-2: Exp 6c (Custom) - EMA + dead-code reset
# ============================================================

set -e
set -o pipefail

if [ -z "${1}" ] || [ -z "${2}" ] || [ -z "${3}" ]; then
  echo "Usage: bash exp_0128/phase3-2/run_exp6c_custom.sh <steps> <dead_code_threshold> <lambda_inter> [inter_warmup_steps] [beta_commit] [n_rvq_layers] [rvq_codebook_size] [ema_usage_penalty]"
  echo "Example (screen): bash exp_0128/phase3-2/run_exp6c_custom.sh 200 2 0.25 0 1.0"
  echo "Example (long):   bash exp_0128/phase3-2/run_exp6c_custom.sh 1000 2 0.25 0 1.0"
  echo "Example (bigger K): bash exp_0128/phase3-2/run_exp6c_custom.sh 200 2 0.5 0 1.0 4 2048"
  echo "Example (usage penalty): bash exp_0128/phase3-2/run_exp6c_custom.sh 200 2 0.5 0 1.0 4 2048 0.02"
  exit 2
fi

STEPS="${1}"
DEAD_CODE_THRESHOLD="${2}"
LAMBDA_INTER="${3}"
INTER_WARMUP_STEPS="${4:-0}"
BETA_COMMIT="${5:-1.0}"
N_RVQ_LAYERS="${6:-4}"
RVQ_CODEBOOK_SIZE="${7:-1024}"
EMA_USAGE_PENALTY="${8:-0.0}"

TH_TAG="${DEAD_CODE_THRESHOLD}"
BETA_TAG="${BETA_COMMIT//./p}"
INTER_TAG="${LAMBDA_INTER//./p}"
K_TAG="${RVQ_CODEBOOK_SIZE}"
UP_TAG="${EMA_USAGE_PENALTY//./p}"

source ~/miniconda3/etc/profile.d/conda.sh
set +e
conda activate test
set -e

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# Prefer 2080 Ti (physical GPU 1). Use cuda:0 inside the process.
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="exp_0128/phase3-2/run_exp6c_custom_steps${STEPS}_L${N_RVQ_LAYERS}_K${K_TAG}_ema_th${TH_TAG}_beta${BETA_TAG}_inter${INTER_TAG}_warm${INTER_WARMUP_STEPS}_up${UP_TAG}_${TIMESTAMP}"

echo "============================================================"
echo "Starting Exp 6c (CUSTOM): EMA + dead-code reset (Phase 3-2)"
echo "============================================================"
echo "steps=${STEPS}"
echo "n_rvq_layers=${N_RVQ_LAYERS}"
echo "rvq_codebook_size=${RVQ_CODEBOOK_SIZE}"
echo "dead_code_threshold=${DEAD_CODE_THRESHOLD}"
echo "beta_commit=${BETA_COMMIT}"
echo "lambda_inter=${LAMBDA_INTER}"
echo "inter_warmup_steps=${INTER_WARMUP_STEPS}"
echo "ema_usage_penalty=${EMA_USAGE_PENALTY}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (process sees cuda:0)"
echo "Output: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "============================================================"

PYTHONUNBUFFERED=1 python exp_0128/phase3/residual_vq/train_rvq_short_run.py \
  --steps ${STEPS} \
  --batch_size 8 \
  --grad_accum 2 \
  --lr 1e-4 \
  --n_rvq_layers ${N_RVQ_LAYERS} \
  --rvq_codebook_size ${RVQ_CODEBOOK_SIZE} \
  --lambda_quant 1.0 \
  --lambda_pre 0.0 \
  --lambda_inter ${LAMBDA_INTER} \
  --beta_commit ${BETA_COMMIT} \
  --lambda_codebook 0.0 \
  --rvq_update ema \
  --ema_decay 0.99 \
  --ema_eps 1e-5 \
  --ema_dead_code_threshold ${DEAD_CODE_THRESHOLD} \
  --ema_usage_penalty ${EMA_USAGE_PENALTY} \
  --inter_warmup_steps ${INTER_WARMUP_STEPS} \
  --early_stop_on_collapse \
  --output_dir ${OUTPUT_DIR} \
  --seed 42 \
  --device cuda:0 \
  --eval_interval 200 \
  |& tee ${OUTPUT_DIR}.log

echo ""
echo "============================================================"
echo "Exp 6c (CUSTOM) Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log saved to: ${OUTPUT_DIR}.log"
echo "============================================================"

#!/bin/bash

# ============================================================
# exp_0128 Phase 3-2: Exp 6a - Quantized Alignment (minimal fix)
# ============================================================

set -e
set -o pipefail

# Optional conda activation (keep consistent with existing run scripts)
source ~/miniconda3/etc/profile.d/conda.sh
set +e
conda activate test
set -e

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# Prefer 2080 Ti (physical GPU 1). Use cuda:0 inside the process.
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="exp_0128/phase3-2/run_exp6a_${TIMESTAMP}"

echo "============================================================"
echo "Starting Exp 6a: Quantized Alignment (Phase 3-2)"
echo "============================================================"
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
  --beta_commit 0.25 \
  --lambda_codebook 1.0 \
  --rvq_update grad \
  --inter_warmup_steps 0 \
  --early_stop_on_collapse \
  --output_dir ${OUTPUT_DIR} \
  --seed 42 \
  --device cuda:0 \
  --eval_interval 200 \
  |& tee ${OUTPUT_DIR}.log

echo ""
echo "============================================================"
echo "Exp 6a Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log saved to: ${OUTPUT_DIR}.log"
echo "============================================================"


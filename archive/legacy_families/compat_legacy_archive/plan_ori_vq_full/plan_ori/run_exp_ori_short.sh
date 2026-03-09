#!/bin/bash
# ============================================================
# exp_0206 Plan Original: Short-run (1000 steps)
# ============================================================
#
# 目的: 驗證 Single VQ 4096 + EMA (pretrained init) 的可行性
#
# 科學問題:
#   1. 預訓練 codebook + EMA 能否避免 collapse？
#   2. Warm start vs Cold start 哪個更好？
#   3. 單層 vs 多層 VQ 的必要性？
#
# 驗收標準:
#   P1 (step 200): top10≤0.95, used≥82, mse≤0.1
#   P2 (step 1000): entropy≥5.0, top10≤0.5, used≥410, mse≤0.1
#   P3 (bonus): entropy>6.5, top10<0.15, used≥2867
#
# 用法:
#   bash families/compat_legacy/plan_ori_vq/plan_ori/run_exp_ori_short.sh [GPU_ID]
# ============================================================

set -e

GPU_ID=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="families/compat_legacy/plan_ori_vq/runs/plan_ori_short_${TIMESTAMP}"

# Conda environment
CONDA_ENV="test"
set +u
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/home/sbplab/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "=========================================="
echo "Exp 0206 - Plan Original: Short-run"
echo "=========================================="
echo "GPU: ${GPU_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Steps: 1000"
echo "Python: $(which python)"
echo "方案: Single VQ K=4096, Pretrained Init, EMA"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

CUDA_VISIBLE_DEVICES=${GPU_ID} python families/compat_legacy/plan_ori_vq/plan_ori/train_single_vq_ema.py \
  --mode step \
  --output_dir "${OUTPUT_DIR}" \
  --steps 1000 \
  --batch_size 8 \
  --grad_accum 2 \
  --learning_rate 1e-4 \
  --eval_interval 200 \
  --checkpoint_interval 200 \
  --eval_max_batches 30 \
  --lambda_quant 1.0 \
  --beta_commit 1.0 \
  --intermediate_weight 0.03 \
  --vq_ema_decay 0.99 \
  --vq_ema_threshold 2 \
  --vq_ema_usage_penalty 0.0 \
  --seed 42

echo ""
echo "=========================================="
echo "✅ Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "分析結果:"
echo "  python families/compat_legacy/plan_ori_vq/plan_ori/analyze_results.py ${OUTPUT_DIR}"
echo "=========================================="

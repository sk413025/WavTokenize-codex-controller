#!/bin/bash
# =============================================================================
# exp_0206: Long-Term RVQ Training (300 Epochs)
#
# 結合 exp_k_v6 epoch-based 訓練 + Phase 3-2 Exp 6c 最佳 RVQ 配置
# 使用方法:
#   bash families/compat_legacy/plan_ori_vq/run.sh        # 使用 GPU 0 (預設)
#   bash families/compat_legacy/plan_ori_vq/run.sh 1      # 使用 GPU 1
#   bash families/compat_legacy/plan_ori_vq/run.sh 2      # 使用 GPU 2
# =============================================================================
set -euo pipefail

# GPU selection
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Using GPU: $GPU_ID"

# Conda environment
CONDA_ENV="test"
echo "Activating conda env: $CONDA_ENV"

# Get the conda base path
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/home/sbplab/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Experiment name
EXP_NAME="${2:-longterm}"

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

python families/compat_legacy/plan_ori_vq/train_long.py \
    --exp_name "$EXP_NAME" \
    --seed 42 \
    \
    --num_epochs 300 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 10 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --use_amp \
    --device cuda:0 \
    \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    \
    --n_rvq_layers 4 \
    --rvq_codebook_size 2048 \
    --ema_decay 0.99 \
    --ema_dead_code_threshold 2 \
    --ema_usage_penalty 0.1 \
    \
    --lambda_quant 1.0 \
    --lambda_pre 0.0 \
    --beta_commit 1.0 \
    --lambda_codebook 0.0 \
    \
    --intermediate_weight 0.5 \
    --intermediate_weight_min 0.25 \
    --warmdown_epochs 50 \
    --intermediate_L3_weight 0.3 \
    --intermediate_L4_weight 0.5 \
    --intermediate_L6_weight 0.5 \
    \
    --curriculum_start 0.3 \
    --curriculum_end 0.85 \
    --curriculum_epochs 200 \
    \
    --save_checkpoint_every 10 \
    --save_audio_interval 50 \
    --eval_max_batches 50 \
    2>&1 | tee -a "families/compat_legacy/plan_ori_vq/runs/${EXP_NAME}_latest.log"

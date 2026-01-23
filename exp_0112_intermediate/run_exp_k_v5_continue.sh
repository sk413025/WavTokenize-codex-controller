#!/bin/bash
# ============================================================
# Exp K v5 Continue: 從 epoch 300 繼續訓練到 500
# ============================================================
#
# 背景:
#   - v5 在 epoch 300 完成，Val Acc 0.845%
#   - Best 是 epoch 141 的 0.899%
#   - 最後 10 epoch 仍有上升趨勢 (+0.005%/epoch)
#   - Train/Val Loss 都在下降，無過擬合
#
# 繼續訓練策略:
#   1. LR: 5e-6 → 1e-6 (cosine decay over 200 epochs)
#      - 原本已達最小值 1e-6，稍微提高到 5e-6 重新開始
#      - 5 epoch warmup
#   2. Curriculum: 固定在 85% (已完成過渡)
#   3. Intermediate Weight: 固定在 0.25 (已完成 warmdown)
#
# 預期:
#   - 按趨勢預測 epoch 500 可達 ~1.0% accuracy
#   - 可能超越 Best (0.899%)
#
# 執行:
#   bash exp_0112_intermediate/run_exp_k_v5_continue.sh
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

CHECKPOINT_DIR="exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="exp_k_v5_continue_${TIMESTAMP}"

echo "============================================================"
echo "Exp K v5 Continue: 301 → 500 epochs"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Experiment: ${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Time: $(date)"
echo ""
echo "Learning Rate: 5e-6 → 1e-6 (cosine decay)"
echo "Curriculum: 85% (fixed)"
echo "Intermediate Weight: 0.25 (fixed)"
echo "============================================================"

python exp_0112_intermediate/train_v5_continue.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --exp_name "${EXP_NAME}" \
    --start_epoch 301 \
    --num_epochs 500 \
    --batch_size 8 \
    --lr 5e-6 \
    --min_lr 1e-6 \
    --warmup_epochs 5 \
    --intermediate_weight 0.25 \
    --curriculum_phase 0.85 \
    --grad_clip 1.0 \
    --use_amp \
    --save_audio_interval 50 \
    2>&1 | tee exp_0112_intermediate/exp_k_v5_continue.log

echo "============================================================"
echo "Exp K v5 Continue Complete!"
echo "============================================================"

#!/bin/bash
# ============================================================
# Exp37: CE Loss Only (基於 Exp35 配置)
# ============================================================
#
# 假設: CE Loss 提供直接的 token 監督，可能更有效
#
# 基於 Exp35 配置，唯一差別:
# - feature_weight: 0.0 (移除 feature)
# - triplet_weight: 0.0 (移除 triplet)
# - ce_weight: 1.0 (純 CE)
# ============================================================

# Activate conda environment with working torchaudio
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1212

echo "============================================================"
echo "Exp37: CE Loss Only (基於 Exp35)"
echo "============================================================"
echo "  - lr: 1e-4, warmup: 10 epochs"
echo "  - lora_rank: 128, dropout: 0.2"
echo "  - feature=0, triplet=0, ce=1.0"
echo "  - epochs: 100"
echo "============================================================"

python train_aligned.py \
    --exp_name exp37_ce_only \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --weight_decay 0.05 \
    --feature_weight 0.0 \
    --triplet_weight 0.0 \
    --triplet_margin 0.2 \
    --ce_weight 1.0 \
    --lr 1e-4 \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --batch_size 16 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 50 \
    2>&1 | tee exp37.log

echo ""
echo "============================================================"
echo "Exp37 completed at: $(date)"
echo "============================================================"

#!/bin/bash
# ============================================================
# Exp48: 最佳配置 + Clean→Clean 過濾
# ============================================================
# 基於 Exp35 的最佳 Loss 配置 (F=1.0, T=1.0, CE=0.0)
# 新增: Clean→Clean 過濾 (已在 data_aligned.py 預設啟用)
# 目標: 驗證修復後的資料是否能進一步提升性能
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217

echo "============================================================"
echo "Exp48: 最佳配置 + Clean→Clean 過濾"
echo "============================================================"
echo "  - Loss: Feature=1.0, Triplet=1.0, CE=0.0 (Exp35 配置)"
echo "  - LoRA: all_18 layers, rank=128"
echo "  - 資料: 過濾 Clean→Clean (28% → 0%)"
echo "  - Epochs: 500"
echo "============================================================"

python train.py \
    --exp_name exp48_best_config \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --batch_size 16 \
    --num_epochs 500 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 50 \
    2>&1 | tee exp48.log

echo "============================================================"
echo "Exp48 completed at: $(date)"
echo "============================================================"

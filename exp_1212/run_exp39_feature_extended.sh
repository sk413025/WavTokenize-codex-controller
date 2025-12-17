#!/bin/bash
# Exp39: Feature Loss Only - Extended Training (Resume from Exp36)
# 從 Exp36 的 100 epochs 繼續訓練到 200 epochs
# 配置與 Exp36 完全一致，僅 warmup=0 (續訓) 和 epochs=200

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1212

echo "============================================================"
echo "Exp39: Feature Loss Extended (Resume from Exp36)"
echo "============================================================"
echo "  - 配置與 Exp36 一致"
echo "  - lora_dropout: 0.2, weight_decay: 0.05"
echo "  - warmup: 0 (續訓), epochs: 200"
echo "============================================================"

python train_aligned.py \
    --exp_name exp39_feature_extended \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --weight_decay 0.05 \
    --feature_weight 1.0 \
    --triplet_weight 0.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --use_scheduler \
    --warmup_epochs 0 \
    --grad_clip 1.0 \
    --batch_size 16 \
    --num_epochs 200 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 50 \
    --resume runs/exp36_feature_only/best_model.pt \
    2>&1 | tee exp39_run.log

echo ""
echo "============================================================"
echo "Exp39 completed at: $(date)"
echo "============================================================"

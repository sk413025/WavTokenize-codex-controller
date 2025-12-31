#!/bin/bash
# Exp73: Denoising Adapter
#
# 核心改進：
# - 在 encoder 後加入專門的去噪 Adapter
# - Adapter 學習 noisy → clean 的殘差映射
# - 使用 LoRA + Adapter 雙重調整

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Exp73: Denoising Adapter"
echo "=============================================="
echo "Adapter Type: mlp"
echo "Adapter Expansion: 4"
echo "Adapter Layers: 2"
echo "=============================================="

python -u exp_1231/train_exp73_adapter.py \
    --exp_name exp73_adapter \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --adapter_type mlp \
    --adapter_expansion 4 \
    --adapter_num_layers 2 \
    --adapter_dropout 0.1 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --soft_token_weight 0.5 \
    --soft_token_temperature 1.0 \
    --vq_commitment_weight 0.1 \
    --vq_distortion_weight 0.1 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --num_epochs 300 \
    --num_workers 4 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 2 \
    --early_stopping_patience 100 \
    --seed 42

echo "=============================================="
echo "Exp73 completed!"
echo "=============================================="

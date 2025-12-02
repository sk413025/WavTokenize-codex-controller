#!/bin/bash
# exp_1201 - Experiment 1: Gumbel-Softmax Distance Loss
#
# 使用 Gumbel-Softmax 實現可微的 Distance Loss
# - Forward: hard codes (argmax + Gumbel noise)
# - Backward: soft gradients (softmax)
#
# 特點：
# - 引入隨機性，幫助探索 codebook 空間
# - 可能逃離局部最優
export CUDA_VISIBLE_DEVICES=1
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201

echo "========================================"
echo "Experiment 1: Gumbel-Softmax Distance Loss"
echo "========================================"
echo ""
echo "Configuration:"
echo "  distance_loss_mode: gumbel"
echo "  temperature: 1.0"
echo "  gumbel_hard: True"
echo "  soft_dist_loss_weight: 0.1"
echo ""

python train.py \
    --exp_name gumbel_baseline \
    --distance_loss_mode gumbel \
    --temperature 1.0 \
    --soft_dist_loss_weight 0.1 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --save_interval 10 \
    --log_interval 50 \
    --num_workers 4

echo ""
echo "Experiment 1 completed!"
echo "Results: experiments/gumbel_baseline/"

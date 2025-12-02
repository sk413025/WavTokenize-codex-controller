#!/bin/bash
# exp_1201 - Experiment 2: STE (Straight-Through Estimator) Distance Loss
#
# 使用 STE 實現可微的 Distance Loss
# - Forward: hard codes (argmax，確定性)
# - Backward: soft gradients (softmax)
#
# 特點：
# - 確定性選擇，訓練更穩定
# - 沒有隨機性，但可能陷入局部最優
export CUDA_VISIBLE_DEVICES=2
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201

echo "========================================"
echo "Experiment 2: STE Distance Loss"
echo "========================================"
echo ""
echo "Configuration:"
echo "  distance_loss_mode: ste"
echo "  temperature: 1.0"
echo "  soft_dist_loss_weight: 0.1"
echo ""

python train.py \
    --exp_name ste_baseline \
    --distance_loss_mode ste \
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
echo "Experiment 2 completed!"
echo "Results: experiments/ste_baseline/"

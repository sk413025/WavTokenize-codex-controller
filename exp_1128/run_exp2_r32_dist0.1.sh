#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# exp_1128 實驗 2: LoRA Rank 32 + Distance Loss 0.1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 與實驗 1 相同 LoRA Rank，但 Distance Loss 更大
# 測試更強的 code alignment 壓力

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1128

export CUDA_VISIBLE_DEVICES=0

nohup python train.py \
    --exp_name lora_r32_dist0.1 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --feature_loss_weight 1.0 \
    --distance_loss_weight 0.1 \
    --vq_loss_weight 0.0 \
    > experiments/lora_r32_dist0.1.log 2>&1 &

echo "Experiment 2 started: LoRA Rank=32, Distance Loss=0.1"
echo "Monitor: tail -f experiments/lora_r32_dist0.1.log"
echo ""
echo "Note: Higher distance loss may lead to:"
echo "  - Better code alignment"
echo "  - But potentially worse feature similarity"

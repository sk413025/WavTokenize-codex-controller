#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# exp_1128 實驗 4: LoRA Rank 64 + Distance Loss 0.1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 最大配置: 大 LoRA + 強 Distance Loss
# 測試上限

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1128

export CUDA_VISIBLE_DEVICES=2

nohup python train.py \
    --exp_name lora_r64_dist0.1 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --feature_loss_weight 1.0 \
    --distance_loss_weight 0.1 \
    --vq_loss_weight 0.0 \
    > experiments/lora_r64_dist0.1.log 2>&1 &

echo "Experiment 4 started: LoRA Rank=64, Distance Loss=0.1"
echo "Monitor: tail -f experiments/lora_r64_dist0.1.log"
echo ""
echo "This is the most aggressive configuration."
echo "Watch for overfitting or training instability."

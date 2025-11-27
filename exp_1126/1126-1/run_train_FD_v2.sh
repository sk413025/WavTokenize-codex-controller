#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1126-1 實驗 - 純 Feature Distillation 背景執行版
# 關閉 Distance Loss，觀察 Token Accuracy 是否保持穩定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1126/1126-1

export CUDA_VISIBLE_DEVICES=0

# 背景執行並記錄 log
nohup python train.py \
    --exp_name lora_encoder_1126_1_FD_v2 \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --feature_loss_weight 1.0 \
    --distance_loss_weight 0.0 \
    --vq_loss_weight 0.0 \
    > experiments/lora_encoder_1126_1_FD_v2.log 2>&1 &

echo "Training started with PURE Feature Distillation!"
echo "Monitor with: tail -f experiments/lora_encoder_1126_1_FD_v2.log"
echo ""
echo "Loss weights:"
echo "  - feature_loss_weight: 1.0"
echo "  - distance_loss_weight: 0.0 (完全關閉)"
echo "  - vq_loss_weight: 0.0"
echo ""
echo "假設：如果 Token Acc 維持穩定，則 Distance Loss 是問題來源"

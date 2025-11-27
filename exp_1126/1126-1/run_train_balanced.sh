#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1126-1 實驗 - 平衡權重版本
# 調整 Distance Loss 權重，使各 loss 貢獻相近
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 根據實驗觀察：
# - Feature Loss ≈ 0.03 → 貢獻 = 0.03 × 1.0 = 0.03
# - Distance Loss ≈ 3.5 → 若權重 0.01，貢獻 = 3.5 × 0.01 = 0.035
# - VQ Loss ≈ 0 → 可忽略
# 
# 這樣兩個 loss 貢獻相當，不會有一個主導

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1126/1126-1

export CUDA_VISIBLE_DEVICES=1

# 背景執行並記錄 log
nohup python train.py \
    --exp_name lora_encoder_1126_1_balanced \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --feature_loss_weight 1.0 \
    --distance_loss_weight 0.01 \
    --vq_loss_weight 0.0 \
    > experiments/lora_encoder_1126_1_balanced.log 2>&1 &

echo "Training started with balanced weights!"
echo "Monitor with: tail -f experiments/lora_encoder_1126_1_balanced.log"
echo ""
echo "Loss weights:"
echo "  - feature_loss_weight: 1.0"
echo "  - distance_loss_weight: 0.01 (降低10倍)"
echo "  - vq_loss_weight: 0.0"

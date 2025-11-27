#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1126-1 實驗 - Frozen VQ v3 版本
# 凍結 Student VQ Codebook，保持與 Teacher 一致
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# v3 方案說明：
# - 問題：之前 quantizer.eval() 或 feature_extractor.eval() 會同時關閉 STE
# - STE (Straight-Through Estimator) 是梯度反傳必需的
# - 方案：保持 training=True，但在每次 forward 後恢復 codebook
# - 效果：EMA 會更新，但立即被恢復，等於凍結 codebook
#
# 預期改善：
# - Token Accuracy 不再崩潰（30% → 3%）
# - Teacher 和 Student 使用相同 codebook
# - Distance Loss 應該能正常收斂

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1126/1126-1

export CUDA_VISIBLE_DEVICES=1

# 背景執行並記錄 log
nohup python train.py \
    --exp_name lora_encoder_frozen_vq_v3 \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --feature_loss_weight 1.0 \
    --distance_loss_weight 0.01 \
    --vq_loss_weight 0.0 \
    > experiments/lora_encoder_frozen_vq_v3.log 2>&1 &

echo "Training started with Frozen VQ v3!"
echo "Monitor with: tail -f experiments/lora_encoder_frozen_vq_v3.log"
echo ""
echo "方案說明:"
echo "  - Codebook 凍結方式: 每次 forward 後恢復"
echo "  - STE 梯度傳遞: 保持正常"
echo "  - LoRA 訓練: 正常"
echo ""
echo "Loss weights:"
echo "  - feature_loss_weight: 1.0"
echo "  - distance_loss_weight: 0.01"
echo "  - vq_loss_weight: 0.0"

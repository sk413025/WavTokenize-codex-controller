#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# exp_1128 實驗 3: LoRA Rank 64 + Distance Loss 0.05
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 更大的 LoRA Rank 提供更多表達能力
# 測試是否需要更多參數才能精確對齊 codes

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1128

export CUDA_VISIBLE_DEVICES=1

nohup python train.py \
    --exp_name lora_r64_dist0.05 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --feature_loss_weight 1.0 \
    --distance_loss_weight 0.05 \
    --vq_loss_weight 0.0 \
    > experiments/lora_r64_dist0.05.log 2>&1 &

echo "Experiment 3 started: LoRA Rank=64, Distance Loss=0.05"
echo "Monitor: tail -f experiments/lora_r64_dist0.05.log"
echo ""
echo "LoRA parameter comparison:"
echo "  - Rank 16: ~38K params"
echo "  - Rank 32: ~77K params"
echo "  - Rank 64: ~154K params"

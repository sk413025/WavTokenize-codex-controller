#!/bin/bash
# ================================================================
# exp9: EmbDistillation + Large LoRA (rank=256)
# ================================================================
#
# 目的:
#   驗證 LoRA 容量假設 - 增加 rank 是否能改善效果
#
# 配置:
#   - lora_rank: 256 (從 64 增加 4 倍)
#   - lora_alpha: 512 (保持 alpha/rank = 2)
#   - Loss: EmbDistillation (比 Feature+VQ 好)
#   - 參數量: 154K → 616K (4x)
#
# Baseline 比較:
#   - exp8 (rank=64): Val Acc 6.76% (比 baseline 4.78% 高 2%)
#   - exp9 (rank=256): 預期 Val Acc 應該更高
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="large_lora_emb"

echo "=========================================="
echo "Running exp9: ${EXP_NAME}"
echo "Strategy: EmbDistillation + Large LoRA (rank=256)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Key changes from exp8:"
echo "  - lora_rank: 64 → 256 (4x)"
echo "  - lora_alpha: 128 → 512"
echo "  - params: 154K → 616K"
echo ""
echo "Expected:"
echo "  - If capacity is the issue: Val Acc should improve significantly"
echo "  - If task is too hard: Val Acc won't improve much"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1203

python train.py \
    --exp_name ${EXP_NAME} \
    --lora_rank 256 \
    --lora_alpha 512 \
    --distance_loss_mode emb_distillation \
    --emb_to_codebook_weight 1.0 \
    --feature_loss_weight 0.0 \
    --soft_dist_loss_weight 0.0 \
    --vq_loss_weight 0.0 \
    --correct_vq_loss_weight 0.0 \
    --ce_token_weight 0.0 \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "Results: experiments/${EXP_NAME}/"
echo ""
echo "Compare with exp8 (rank=64):"
echo "  - exp8 Val Acc: 6.76%"
echo "  - exp9 Val Acc: check experiments/${EXP_NAME}/training_history.json"
echo "=========================================="

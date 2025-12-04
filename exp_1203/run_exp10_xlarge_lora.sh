#!/bin/bash
# ================================================================
# exp10: EmbDistillation + Medium LoRA (rank=128) + Large Batch (32)
# ================================================================
#
# 目的:
#   測試假設：Batch Size 比 LoRA 容量更重要
#
# 配置:
#   - lora_rank: 128 (中等容量)
#   - lora_alpha: 256 (保持 alpha/rank = 2)
#   - batch_size: 32 (比 exp9 的 16 大)
#   - Loss: EmbDistillation
#   - 參數量: ~308K
#
# 對比:
#   - exp9 (rank=256, batch=16): Val Acc ~8.83%
#   - exp10_old (rank=512, batch=8): Val Acc ~7-8% (更差)
#   - exp10_new (rank=128, batch=32): ?
#
# 預期:
#   - 如果 batch size 更重要: Val Acc 應該 > 8.83%
#   - 如果 LoRA 容量更重要: Val Acc 應該 < 8.83%
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=0

# 實驗名稱
EXP_NAME="medium_lora_large_batch"

echo "=========================================="
echo "Running exp10: ${EXP_NAME}"
echo "Strategy: EmbDistillation + Medium LoRA (rank=128) + Large Batch (32)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Key hypothesis:"
echo "  - Batch Size 比 LoRA 容量更重要"
echo ""
echo "Config:"
echo "  - lora_rank: 128"
echo "  - lora_alpha: 256"
echo "  - batch_size: 32"
echo ""
echo "Comparison:"
echo "  - exp9 (rank=256, batch=16): Val Acc ~8.83%"
echo "  - exp10_old (rank=512, batch=8): worse"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1203

python train.py \
    --exp_name ${EXP_NAME} \
    --lora_rank 128 \
    --lora_alpha 256 \
    --distance_loss_mode emb_distillation \
    --emb_to_codebook_weight 1.0 \
    --feature_loss_weight 0.0 \
    --soft_dist_loss_weight 0.0 \
    --vq_loss_weight 0.0 \
    --correct_vq_loss_weight 0.0 \
    --ce_token_weight 0.0 \
    --batch_size 20 \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "Results: experiments/${EXP_NAME}/"
echo ""
echo "Compare with previous experiments:"
echo "  - exp9 (rank=256, batch=16):  Val Acc ~8.83%"
echo "  - exp10 (rank=128, batch=32): check experiments/${EXP_NAME}/training_history.json"
echo ""
echo "If Val Acc > 8.83%: Batch size is more important than LoRA capacity"
echo "=========================================="

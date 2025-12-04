#!/bin/bash
# ================================================================
# exp10: EmbDistillation + XLarge LoRA (rank=512)
# ================================================================
#
# 目的:
#   繼續驗證 LoRA 容量假設 - 從 exp9 (rank=256) 再增加
#
# 配置:
#   - lora_rank: 512 (從 256 增加 2 倍)
#   - lora_alpha: 1024 (保持 alpha/rank = 2)
#   - Loss: EmbDistillation
#   - 參數量: 616K → 1.2M (2x)
#
# 新增:
#   - 包含 VQ 後 feature loss 監控 (monitor/vq_feature_loss)
#   - 可在 TensorBoard 觀察 VQ 後特徵是否也在接近
#
# Baseline 比較:
#   - exp8 (rank=64):  Val Acc ~9.99%
#   - exp9 (rank=256): Val Acc ~9.99%
#   - exp10 (rank=512): ?
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="xlarge_lora_emb"

echo "=========================================="
echo "Running exp10: ${EXP_NAME}"
echo "Strategy: EmbDistillation + XLarge LoRA (rank=512)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Key changes from exp9:"
echo "  - lora_rank: 256 → 512 (2x)"
echo "  - lora_alpha: 512 → 1024"
echo "  - params: 616K → 1.2M"
echo ""
echo "New monitoring:"
echo "  - monitor/vq_feature_loss: VQ 後特徵差異 (不參與訓練)"
echo ""
echo "Expected:"
echo "  - If capacity is still the issue: Val Acc should improve"
echo "  - If capacity is saturated: Val Acc won't improve"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1203

python train.py \
    --exp_name ${EXP_NAME} \
    --lora_rank 512 \
    --lora_alpha 1024 \
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
echo "Compare with previous experiments:"
echo "  - exp8 (rank=64):   Val Acc ~9.99%"
echo "  - exp9 (rank=256):  Val Acc ~9.99%"
echo "  - exp10 (rank=512): check experiments/${EXP_NAME}/training_history.json"
echo ""
echo "Check TensorBoard for:"
echo "  - monitor/vq_feature_loss: VQ 後特徵差異趨勢"
echo "=========================================="

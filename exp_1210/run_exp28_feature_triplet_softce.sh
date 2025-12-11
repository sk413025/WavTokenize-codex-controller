#!/bin/bash
# ============================================================
# Exp28: Feature + Triplet + Soft CE (知識蒸餾綜合方案)
# ============================================================
#
# 目的: 結合 Triplet 對比學習 + Soft CE 知識蒸餾
#
# 理論基礎:
# - Triplet Loss: 拉近正樣本、推遠負樣本 (對比學習)
# - Soft CE Loss: 使用 Teacher 的 softmax 分布作為軟目標
#   - 保留更多梯度資訊 (相比硬標籤)
#   - 經典知識蒸餾方法，理論基礎紮實
#
# Loss 配置:
# - Feature MSE: 1.0 (特徵對齊)
# - Triplet: 0.3 (對比學習，略降權重避免衝突)
# - Soft CE: 0.5 (知識蒸餾)
# - DW: 0.0 (關閉)
#
# 預期:
# - Triplet + Soft CE 互補：
#   - Triplet 優化特徵空間結構
#   - Soft CE 優化 token 分布對齊
# - Val Token Acc 應優於 Exp27 (純 Triplet)
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

echo "============================================================"
echo "Exp28: Feature + Triplet + Soft CE"
echo "============================================================"
echo "Loss configuration:"
echo "  - Feature MSE: 1.0"
echo "  - Triplet: 0.3 (margin=0.2)"
echo "  - Soft CE: 0.5 (temperature=2.0)"
echo "============================================================"

python train_lora_v3.py \
    --exp_name exp28_feature_triplet_softce \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_weight 1.0 \
    --triplet_weight 0.3 \
    --triplet_margin 0.2 \
    --soft_ce_weight 0.5 \
    --soft_ce_temperature 2.0 \
    --dw_weight 0.0 \
    --lr 2e-5 \
    --batch_size 14 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp28.log

echo ""
echo "============================================================"
echo "Exp28 completed at: $(date)"
echo "============================================================"

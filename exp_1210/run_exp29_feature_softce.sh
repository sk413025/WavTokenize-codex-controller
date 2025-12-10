#!/bin/bash
# ============================================================
# Exp29: Feature + Soft CE (純知識蒸餾方案)
# ============================================================
#
# 目的: 測試純知識蒸餾 (無 Triplet) 的效果
#
# 理論基礎:
# - Soft CE Loss (KL Divergence):
#   - 使用 Teacher 的 softmax 分布作為軟目標
#   - 比硬標籤 (one-hot) 保留更多資訊
#   - Hinton 經典知識蒸餾方法
#
# - Temperature 參數:
#   - T=2.0: 軟化分布，讓模型學習更平滑的分布
#   - 較高 T 讓 "dark knowledge" 更容易傳遞
#
# Loss 配置:
# - Feature MSE: 1.0 (特徵對齊)
# - Triplet: 0.0 (關閉，與 Exp28 對比)
# - Soft CE: 1.0 (知識蒸餾，主要監督信號)
# - DW: 0.0 (關閉)
#
# 預期:
# - 與 Exp26 (Feature Only) 對比: Soft CE 應提升 Val Acc
# - 與 Exp28 (Feature+Triplet+SoftCE) 對比: 確認 Triplet 的貢獻
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

echo "============================================================"
echo "Exp29: Feature + Soft CE (Pure Knowledge Distillation)"
echo "============================================================"
echo "Loss configuration:"
echo "  - Feature MSE: 1.0"
echo "  - Triplet: 0.0 (disabled)"
echo "  - Soft CE: 1.0 (temperature=2.0)"
echo "============================================================"

python train_lora_v3.py \
    --exp_name exp29_feature_softce \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_weight 1.0 \
    --triplet_weight 0.0 \
    --triplet_margin 0.2 \
    --soft_ce_weight 1.0 \
    --soft_ce_temperature 2.0 \
    --dw_weight 0.0 \
    --lr 2e-5 \
    --batch_size 16 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp29.log

echo ""
echo "============================================================"
echo "Exp29 completed at: $(date)"
echo "============================================================"

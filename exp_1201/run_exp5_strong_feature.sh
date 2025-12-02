#!/bin/bash
# exp_1201/run_exp5_strong_feature.sh
#
# 實驗目的: 組合策略 - 強 Feature Loss + CE Token Loss
# 動機: 解決特徵「沒有貼在一起」的問題
#
# 問題分析 (來自 t-SNE 分析):
#   - L2 Distance: 0.863 → 0.840 (僅 -2.7%)
#   - 特徵對齊「飽和」但沒有真正收斂到 0
#   - Cosine Similarity ~0.94 方向相似，但距離不夠近
#   - 需要更強的「拉力」把 Student 特徵拉到 Teacher 上
#
# 配置策略:
#   - feature_loss_weight: 5.0 (大幅提高！原本 1.0)
#   - soft_dist_loss_weight: 1.0 (CE Loss 權重也提高，原本 0.5)
#   - temperature: 0.5 (更銳利的分布)
#   - learning_rate: 5e-5 (略低，避免不穩定)
#
# 與之前實驗的對比:
#   - exp3 (CE): feature_weight=1.0, dist_weight=0.5
#   - exp5 (本實驗): feature_weight=5.0, dist_weight=1.0
#
# 理論基礎:
#   1. 強 Feature Loss → 強迫 Student 特徵「貼到」Teacher
#   2. CE Loss → 直接優化 Token Accuracy
#   3. 兩者互補：特徵接近 + Token 選對
#
# 預期結果:
#   - L2 Distance 大幅下降 (目標 < 0.5)
#   - Token Accuracy 顯著提升 (目標 > 20%)

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="strong_feature_ce"

echo "=========================================="
echo "Running exp_1201: ${EXP_NAME}"
echo "Strategy: Strong Feature Loss + CE Token Loss"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Problem to solve:"
echo "  - Features are NOT converging to Teacher"
echo "  - L2 Distance stuck at ~0.84 (should be ~0)"
echo ""
echo "Key changes from exp3:"
echo "  - feature_loss_weight: 1.0 -> 5.0 (5x stronger!)"
echo "  - soft_dist_loss_weight: 0.5 -> 1.0 (2x stronger)"
echo "  - learning_rate: 1e-4 -> 5e-5 (for stability)"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201

python train.py \
    --exp_name ${EXP_NAME} \
    --distance_loss_mode ce \
    --feature_loss_weight 5.0 \
    --soft_dist_loss_weight 1.0 \
    --temperature 0.5 \
    --label_smoothing 0.05 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "=========================================="

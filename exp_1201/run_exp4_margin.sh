#!/bin/bash
# exp_1201/run_exp4_margin.sh
#
# 實驗目的: 使用 Margin Loss 優化決策邊界
# 動機: 確保正確 token 與錯誤 token 有足夠的距離差
#
# 方法:
#   - Margin Loss: max(0, d(x, correct) - d(x, wrong) + margin)
#   - 不只要接近正確 token，還要遠離錯誤 token
#   - 結合 Feature Loss 保持特徵連續性
#
# Codebook 距離統計 (來自 wavtok_distance_mat_corrected.pt):
#   - 最近鄰平均距離: 1.42
#   - 中位距離: 5.32
#   - 標準差: 4.56
#
# 配置策略:
#   - margin: 0.5 (< 最近鄰距離 1.42，確保可學習)
#   - soft_dist_loss_weight: 0.3 (避免 margin loss 主導)
#   - 較大 batch_size: 16 (更穩定的 hard negative 採樣)
#
# 與 CE 的對比:
#   - CE: 所有錯誤 token 同等懲罰
#   - Margin: 只關注最近的錯誤 token (hard negative)
#   - Margin 理論上對 codebook 的幾何結構更友好
#
# 預期結果:
#   - Token Accuracy 穩定提升
#   - margin_satisfied_rate 逐漸增加

# 設定 GPU
export CUDA_VISIBLE_DEVICES=2

# 實驗名稱
EXP_NAME="margin_tuned"

echo "=========================================="
echo "Running exp_1201: ${EXP_NAME}"
echo "Distance Loss Mode: Margin (Tuned)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Codebook distance stats:"
echo "  - Avg nearest neighbor: 1.42"
echo "  - Median distance: 5.32"
echo ""
echo "Key settings:"
echo "  - margin: 0.5 (learnable, < nearest neighbor dist)"
echo "  - soft_dist_loss_weight: 0.3 (balanced)"
echo "  - batch_size: 16"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201

python train.py \
    --exp_name ${EXP_NAME} \
    --distance_loss_mode margin \
    --feature_loss_weight 1.0 \
    --soft_dist_loss_weight 0.3 \
    --margin 0.5 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "=========================================="

#!/bin/bash
# exp_1201/run_exp3_ce.sh
#
# 實驗目的: 使用 Cross-Entropy Loss 直接監督 Token 選擇
# 動機: Token Accuracy 低的本質問題是 soft distance loss 不直接優化 token 選擇
#
# 歷史經驗 (來自 Exp3-2 分支):
#   - Standard CE 在 zero-shot denoising 達到 48.16% Val Acc
#   - Weighted CE + Geometric MSE 反而欠擬合 (44.61%)
#   - Label smoothing 可能有幫助，但不要太激進
#
# 本實驗特殊考量:
#   - exp_1201 是 noisy→clean 的 encoder denoising 任務
#   - 比 Exp3 的 speaker-conditioned 任務更難
#   - 需要同時優化 feature alignment 和 token prediction
#
# 配置策略:
#   - temperature: 0.5 (適中，不要太尖銳導致梯度爆炸)
#   - soft_dist_loss_weight: 0.5 (與 feature loss 平衡)
#   - label_smoothing: 0.05 (輕微，防止過擬合但不削弱學習)
#   - 保持 1e-4 LR (CE 梯度穩定時可以用較大 LR)
#
# 預期結果:
#   - Token Accuracy 提升到 5-15% (相比 STE 的 2%)
#   - 如果 CE loss 收斂，再嘗試增加權重

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="ce_balanced"

echo "=========================================="
echo "Running exp_1201: ${EXP_NAME}"
echo "Distance Loss Mode: Cross-Entropy (Balanced)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Based on Exp3-2 branch learnings:"
echo "  - Standard CE worked (48.16% Val Acc)"
echo "  - Weighted CE caused underfitting"
echo ""
echo "Key settings:"
echo "  - temperature: 0.5 (moderate sharpness)"
echo "  - soft_dist_loss_weight: 0.5 (balanced)"
echo "  - label_smoothing: 0.05 (light regularization)"
echo "  - learning_rate: 1e-4"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201

python train.py \
    --exp_name ${EXP_NAME} \
    --distance_loss_mode ce \
    --feature_loss_weight 1.0 \
    --soft_dist_loss_weight 0.5 \
    --temperature 0.5 \
    --label_smoothing 0.05 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "=========================================="

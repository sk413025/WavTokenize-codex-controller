#!/bin/bash
# exp_1201/run_exp7_vq_only.sh
#
# ================================================================
# 實驗目的: 關閉 Distance Loss，改用 VQ Loss 監督離散 token
# ================================================================
#
# 動機 (來自 strong_feature_ce 的觀察):
#   圖表顯示矛盾現象：
#   1. Feature Loss 下降 ✓ (連續特徵對齊正在學習)
#   2. Distance Loss 下降 (訓練集) ✓
#   3. VQ Loss 上升 🔴 (異常！)
#   4. Token Accuracy 從 35% 暴跌到 ~7% 🔴
#
#   結論：Distance Loss 可能有邏輯問題
#   - soft_codes 分布太平滑
#   - 加權距離看起來在下降，但實際 argmax 選錯
#   - Distance Loss 和 Token Accuracy 的優化方向不一致
#
# 假設:
#   - VQ Loss (commitment loss) 直接監督離散 token 選擇
#   - 關閉 distance loss，改用 VQ loss 應該更直接影響 Token Accuracy
#
# 配置:
#   - distance_loss_mode: ce (保持 CE 結構，但權重設為 0)
#   - soft_dist_loss_weight: 0.0 (關閉 distance loss!)
#   - vq_loss_weight: 100.0 (開啟 VQ loss! 需要大權重因為 VQ Loss ~0.001)
#   - feature_loss_weight: 5.0 (保持強特徵對齊)
#
# 權重計算:
#   Feature Loss ≈ 0.030, weight=5.0 → 貢獻 ~0.15
#   VQ Loss ≈ 0.0014, weight=100.0 → 貢獻 ~0.14 (平衡!)
#
# 注意: Val VQ Loss = 0 是正常的 (VQ 在 eval 模式不計算 commitment loss)
#
# 對照組:
#   - exp5 (strong_feature_ce): dist_weight=1.0, vq_weight=0.0
#   - exp7 (本實驗): dist_weight=0.0, vq_weight=100.0
#
# 預期結果:
#   - 如果假設正確: Token Accuracy 應該上升（或至少不再暴跌）
#   - 如果假設錯誤: Token Accuracy 繼續下降，說明問題不在 distance loss
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="vq_only"

echo "=========================================="
echo "Running exp_1201: ${EXP_NAME}"
echo "Strategy: Feature Loss + VQ Loss (NO Distance Loss)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Hypothesis:"
echo "  - Distance Loss may have design issues"
echo "  - soft_codes distribution too smooth"
echo "  - VQ Loss directly supervises discrete tokens"
echo ""
echo "Key changes from exp5 (strong_feature_ce):"
echo "  - soft_dist_loss_weight: 1.0 -> 0.0 (DISABLED!)"
echo "  - vq_loss_weight: 0.0 -> 100.0 (ENABLED! Large weight needed)"
echo "  - feature_loss_weight: 5.0 (same)"
echo ""
echo "Weight balance:"
echo "  - Feature: 0.030 * 5.0 = 0.15"
echo "  - VQ: 0.0014 * 100.0 = 0.14"
echo ""
echo "Expected:"
echo "  - Token Accuracy should NOT collapse"
echo "  - VQ Loss should guide discrete alignment"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201

python train.py \
    --exp_name ${EXP_NAME} \
    --distance_loss_mode ce \
    --feature_loss_weight 5.0 \
    --soft_dist_loss_weight 0.0 \
    --vq_loss_weight 100.0 \
    --temperature 1.0 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --save_interval 10 \
    --log_interval 50 \
    --num_workers 4

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "Results: experiments/${EXP_NAME}/"
echo ""
echo "Compare with exp5 (strong_feature_ce) to verify hypothesis:"
echo "  - If Token Acc improves: Distance Loss was the problem"
echo "  - If Token Acc still drops: Look elsewhere"
echo "=========================================="

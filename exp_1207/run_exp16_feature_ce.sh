#!/bin/bash
# exp16: Feature Loss + Cross-Entropy Loss
#
# 核心洞察（來自 exp_1204 診斷）：
#   - mean_correct_distance: 3.75  (到正確 token 的距離)
#   - mean_min_distance: 0.45      (到最近錯誤 token 的距離)
#   - 問題：MSE 讓特徵「靠近」但不保證「跨過 Voronoi 邊界」
#
# 解決方案：
#   - Feature Loss: 確保特徵結構穩定（不要崩壞）
#   - CE Loss: 強迫特徵跨回正確的 Voronoi Cell
#
# 使用 GPU 0 (GTX 1080 Ti, 11GB)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "exp16: Feature Loss + Cross-Entropy Loss"
echo "============================================================"
echo ""
echo "診斷發現："
echo "  - Student 特徵離『最近錯誤 token』比『正確 token』更近"
echo "  - MSE Loss 無法解決這個問題"
echo "  - CE Loss 提供強梯度，強迫跨過 Voronoi 邊界"
echo ""

# ============================================================
# 實驗 16a: Feature + CE (equal weight)
# ============================================================
echo "============================================================"
echo "exp16a: Feature + CE (λ_feature=1.0, λ_ce=1.0)"
echo "============================================================"

python train_with_ce.py \
    --exp_name exp16a_feature_ce_equal \
    --num_epochs 30 \
    --batch_size 28 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --feature_weight 1.0 \
    --ce_weight 1.0 \
    --ce_temperature 0.1 \
    --save_interval 10 \
    --plot_interval 10 \
    --audio_interval 10

echo ""
echo "exp16a 完成！"
echo ""

# ============================================================
# 實驗 16b: Feature + CE (CE dominant)
# ============================================================
echo "============================================================"
echo "exp16b: Feature + CE (λ_feature=0.1, λ_ce=1.0)"
echo "============================================================"

python train_with_ce.py \
    --exp_name exp16b_feature_ce_dominant \
    --num_epochs 30 \
    --batch_size 28 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --feature_weight 0.1 \
    --ce_weight 1.0 \
    --ce_temperature 0.1 \
    --save_interval 10 \
    --plot_interval 10 \
    --audio_interval 10

echo ""
echo "exp16b 完成！"
echo ""

# ============================================================
# 實驗 16c: Pure CE (no feature loss)
# ============================================================
echo "============================================================"
echo "exp16c: Pure CE (λ_feature=0.0, λ_ce=1.0)"
echo "============================================================"

python train_with_ce.py \
    --exp_name exp16c_pure_ce \
    --num_epochs 30 \
    --batch_size 28 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --feature_weight 0.0 \
    --ce_weight 1.0 \
    --ce_temperature 0.1 \
    --save_interval 10 \
    --plot_interval 10 \
    --audio_interval 10

echo ""
echo "exp16c 完成！"
echo ""

# ============================================================
# 總結
# ============================================================
echo "============================================================"
echo "所有 exp16 實驗完成！"
echo "============================================================"
echo ""
echo "結果位置:"
echo "  - experiments/exp16a_feature_ce_equal/"
echo "  - experiments/exp16b_feature_ce_dominant/"
echo "  - experiments/exp16c_pure_ce/"
echo ""
echo "預期結果："
echo "  - 如果 CE Loss 有效，Token Accuracy 應該提升"
echo "  - Feature Loss 防止特徵空間崩壞"
echo "  - 最佳配置可能是 exp16a 或 exp16b"

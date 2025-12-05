#!/bin/bash
# ================================================================
# exp13: Linear + CE Loss (方案 A)
# ================================================================
#
# 核心改進：
#   - 不經過 codebook 距離計算
#   - 直接用 Linear(512 → 4096) + Cross-Entropy
#   - Loss 優化目標和 Token Accuracy 完全一致
#
# 診斷結果顯示：
#   - MSE Loss 只讓 student「接近」target，不保證「最近」
#   - 到正確 token 距離 = 3.75，到最近 token 距離 = 0.45
#   - 這導致 Token Accuracy 只有 2.21%
#
# 預期效果：
#   - Token Accuracy 大幅提升（目標 > 50%）
#   - 訓練穩定（純 CE Loss，數值穩定）
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=0

# 實驗名稱
EXP_NAME="exp13_linear_ce"

echo "=========================================="
echo "Running exp_1205: ${EXP_NAME}"
echo "Strategy: Linear projection + Cross-Entropy"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Key insight from diagnosis:"
echo "  - MSE Loss: student gets 'close' but not 'closest'"
echo "  - Distance to correct: 3.75, to nearest: 0.45"
echo "  - Linear+CE directly optimizes Token Accuracy"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1205

python train.py \
    --exp_name ${EXP_NAME} \
    --loss_type linear_ce \
    --label_smoothing 0.1 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --batch_size 20 \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "Results: experiments/${EXP_NAME}/"
echo ""
echo "Compare with exp_1204:"
echo "  - exp11 (MSE + CE, distance-based): Val Acc ~2%"
echo "  - exp13 (Linear + CE, direct):      check experiments/${EXP_NAME}/"
echo ""
echo "If Val Acc >> 2%: Linear+CE successfully bypasses the distance problem!"
echo "=========================================="

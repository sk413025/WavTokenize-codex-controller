#!/bin/bash
# ================================================================
# exp14: Margin-based Loss (方案 B)
# ================================================================
#
# 核心改進：
#   - 不只要求「接近」正確 token
#   - 還要求比最近的錯誤 token「更近」一個 margin
#   - loss = max(0, dist_correct - dist_wrong + margin)
#
# 診斷結果顯示：
#   - MSE Loss 只優化「接近」，不保證「最近」
#   - Margin Loss 強制模型區分正確和錯誤 token
#
# 預期效果：
#   - Token Accuracy 提升
#   - 學習更有判別性的 embedding
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="exp14_margin"

echo "=========================================="
echo "Running exp_1205: ${EXP_NAME}"
echo "Strategy: Margin-based Loss"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Key insight from diagnosis:"
echo "  - MSE doesn't care about 'nearest', only 'close'"
echo "  - Margin Loss: correct must be closer than wrong by margin"
echo "  - margin = 0.5"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1205

python train.py \
    --exp_name ${EXP_NAME} \
    --loss_type margin \
    --margin 0.5 \
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
echo "  - exp11 (MSE + CE):       Val Acc ~2%"
echo "  - exp14 (Margin Loss):    check experiments/${EXP_NAME}/"
echo ""
echo "If Val Acc improves: Margin Loss helps discriminate tokens!"
echo "=========================================="

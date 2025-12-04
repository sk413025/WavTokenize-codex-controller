#!/bin/bash
# ================================================================
# exp_1204: Curriculum Learning + Mixed Loss (MSE + CE)
# ================================================================
#
# 核心改進：
#   1. 漸進式學習 (Curriculum Learning):
#      - 階段 1 (epoch 0-5): 高溫度 MSE 為主，CE=0
#      - 階段 2 (epoch 5-25): 降低溫度，漸進增加 CE 權重
#      - 階段 3 (epoch 25+): 低溫度，CE 達到最終權重
#
#   2. 混合 Loss:
#      - MSE Loss: 穩定訓練，提供平滑梯度
#      - CE Loss: 直接監督 token 選擇，強烈梯度
#
#   3. Temperature Annealing:
#      - 初始: temperature=2.0（軟監督）
#      - 最終: temperature=0.1（硬監督）
#
# 預期效果：
#   - Token Accuracy 從 ~10% 提升到更高
#   - 訓練更穩定（先 MSE 打底，再 CE 精調）
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="curriculum_mse_ce"

echo "=========================================="
echo "Running exp_1204: ${EXP_NAME}"
echo "Strategy: Curriculum Learning + Mixed Loss (MSE + CE)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Curriculum Schedule:"
echo "  - Epoch 0-5: warm-up (MSE only, temp=2.0)"
echo "  - Epoch 5-25: transition (MSE + CE, temp↓)"
echo "  - Epoch 25+: refinement (MSE + CE, temp=0.1)"
echo ""
echo "Loss Weights:"
echo "  - MSE weight: 1.0"
echo "  - CE weight (final): 0.5"
echo "  - Temperature: 2.0 → 0.1"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1204

python train.py \
    --exp_name ${EXP_NAME} \
    --lora_rank 128 \
    --lora_alpha 256 \
    --use_curriculum true \
    --mse_weight 1.0 \
    --ce_weight 0.5 \
    --initial_temperature 2.0 \
    --final_temperature 0.1 \
    --curriculum_mode linear \
    --warmup_epochs 5 \
    --transition_epochs 20 \
    --feature_loss_weight 0.0 \
    --batch_size 20 \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "Results: experiments/${EXP_NAME}/"
echo ""
echo "Compare with exp_1203 experiments:"
echo "  - exp10 (MSE only, no curriculum): Val Acc ~10%"
echo "  - exp_1204 (MSE+CE, curriculum): check experiments/${EXP_NAME}/"
echo ""
echo "If Val Acc improves: Curriculum + CE helps!"
echo "=========================================="

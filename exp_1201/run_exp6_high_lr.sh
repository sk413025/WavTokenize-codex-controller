#!/bin/bash
# exp_1201/run_exp6_high_lr.sh
#
# 實驗目的: 測試更高的 LoRA 學習率對特徵對齊的影響
# 基準: exp2 (STE baseline)
#
# 動機:
#   - exp2 使用 5e-5 學習率，特徵對齊飽和在 L2 ~0.84
#   - 假設：學習率太低，LoRA 無法充分更新
#   - 測試：將學習率提高 10x 到 5e-4
#
# 配置 (基於 exp2):
#   - distance_loss_mode: ste (與 exp2 相同)
#   - temperature: 1.0 (與 exp2 相同)
#   - soft_dist_loss_weight: 0.1 (與 exp2 相同)
#   - learning_rate: 5e-4 (原本 5e-5，提高 10x)
#
# 風險:
#   - 學習率太高可能導致訓練不穩定
#   - 可能需要調整 grad_clip
#
# 預期結果:
#   - 如果成功：L2 Distance 更快下降
#   - 如果失敗：Loss 震盪或發散，需要降低 LR

# 設定 GPU
export CUDA_VISIBLE_DEVICES=2

# 實驗名稱
EXP_NAME="ste_high_lr"

echo "=========================================="
echo "Running exp_1201: ${EXP_NAME}"
echo "Distance Loss Mode: STE (High Learning Rate)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Based on exp2 (STE baseline), key changes:"
echo "  - learning_rate: 5e-5 -> 5e-4 (10x higher!)"
echo ""
echo "Hypothesis:"
echo "  - Low LR causes feature alignment saturation"
echo "  - Higher LR may enable better convergence"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1201

python train.py \
    --exp_name ${EXP_NAME} \
    --distance_loss_mode ste \
    --temperature 1.0 \
    --soft_dist_loss_weight 0.1 \
    --feature_loss_weight 1.0 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --save_interval 10 \
    --log_interval 50 \
    --num_workers 4

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "Results: experiments/${EXP_NAME}/"
echo "=========================================="

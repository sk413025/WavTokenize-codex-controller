#!/bin/bash
# ================================================================
# exp11: MSE + CE (無 Curriculum)
# ================================================================
#
# 消融實驗：測試 CE Loss 單獨效果
#
# 設定：
#   - MSE weight: 1.0
#   - CE weight: 0.5 (固定，無漸進)
#   - Temperature: 1.0 (固定，無 annealing)
#   - 無 curriculum learning
#
# 對比：
#   - exp_1204 (MSE + CE + Curriculum): 完整方案
#   - exp11 (MSE + CE): 只加 CE，無 curriculum
#   - exp12 (MSE + Temp Annealing): 只加溫度漸進
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="exp11_ce_only"

echo "=========================================="
echo "Running exp11: ${EXP_NAME}"
echo "Strategy: MSE + CE (No Curriculum)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Key settings:"
echo "  - MSE weight: 1.0"
echo "  - CE weight: 0.5 (fixed)"
echo "  - Temperature: 1.0 (fixed)"
echo "  - Curriculum: OFF"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1204

python train.py \
    --exp_name ${EXP_NAME} \
    --lora_rank 128 \
    --lora_alpha 256 \
    --use_curriculum false \
    --mse_weight 1.0 \
    --ce_weight 0.5 \
    --initial_temperature 1.0 \
    --final_temperature 1.0 \
    --curriculum_mode linear \
    --warmup_epochs 0 \
    --transition_epochs 1 \
    --feature_loss_weight 0.0 \
    --batch_size 20 \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo ""
echo "Ablation Study Results:"
echo "  - exp_1204 (MSE+CE+Curriculum): check curriculum_mse_ce/"
echo "  - exp11 (MSE+CE only): check ${EXP_NAME}/"
echo "  - exp12 (MSE+TempAnnealing): pending"
echo ""
echo "If exp11 > exp_1204: CE alone is enough"
echo "If exp11 < exp_1204: Curriculum helps"
echo "=========================================="

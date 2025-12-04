#!/bin/bash
# ================================================================
# exp12: MSE + Temperature Annealing (無 CE)
# ================================================================
#
# 消融實驗：測試 Temperature Annealing 單獨效果
#
# 設定：
#   - MSE weight: 1.0
#   - CE weight: 0.0 (不使用 CE)
#   - Temperature: 2.0 → 0.1 (annealing)
#   - Curriculum: ON (只控制溫度)
#
# 對比：
#   - exp_1204 (MSE + CE + Curriculum): 完整方案
#   - exp11 (MSE + CE): 只加 CE，無 curriculum
#   - exp12 (MSE + Temp Annealing): 只加溫度漸進，無 CE
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=2

# 實驗名稱
EXP_NAME="exp12_temp_annealing"

echo "=========================================="
echo "Running exp12: ${EXP_NAME}"
echo "Strategy: MSE + Temperature Annealing (No CE)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Key settings:"
echo "  - MSE weight: 1.0"
echo "  - CE weight: 0.0 (OFF)"
echo "  - Temperature: 2.0 → 0.1"
echo "  - Curriculum: ON (temp only)"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1204

python train.py \
    --exp_name ${EXP_NAME} \
    --lora_rank 128 \
    --lora_alpha 256 \
    --use_curriculum true \
    --mse_weight 1.0 \
    --ce_weight 0.0 \
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
echo ""
echo "Ablation Study Results:"
echo "  - exp_1204 (MSE+CE+Curriculum): check curriculum_mse_ce/"
echo "  - exp11 (MSE+CE only): check exp11_ce_only/"
echo "  - exp12 (MSE+TempAnnealing): check ${EXP_NAME}/"
echo ""
echo "If exp12 > exp11: Temperature Annealing helps more"
echo "If exp12 < exp11: CE helps more"
echo "=========================================="

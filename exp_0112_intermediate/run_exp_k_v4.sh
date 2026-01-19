#!/bin/bash
# ============================================================
# Exp K v4: Optimized Intermediate Layer Supervision
# ============================================================
#
# 改進重點:
#   1. 移除 L10 監督 (效果存疑，loss 僅佔 0.015%)
#   2. L5 權重提高到 1.0 (收斂最好，val_loss 0.49)
#   3. L6 權重降低到 0.5 (避免過擬合)
#   4. 總權重降低到 0.5
#   5. weight_decay 提高到 0.1 (正則化)
#
# 執行:
#   bash exp_0112_intermediate/run_exp_k_v4.sh
# ============================================================

set -e

# 設定環境
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 實驗名稱與時間戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="exp_k_v4_${TIMESTAMP}"

echo "============================================================"
echo "Exp K v4: Optimized Intermediate Layer Supervision"
echo "============================================================"
echo "實驗名稱: ${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo ""
echo "V4 改進:"
echo "  - 移除 L10 監督 (效果存疑)"
echo "  - L3: 0.3, L5: 1.0 (best), L6: 0.5"
echo "  - intermediate_weight: 0.5 (降低總權重)"
echo "  - weight_decay: 0.1 (增強正則化)"
echo "============================================================"

python exp_0112_intermediate/train_v4.py \
    --exp_name "${EXP_NAME}" \
    --seed 42 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --intermediate_weight 0.5 \
    --intermediate_L3_weight 0.3 \
    --intermediate_L5_weight 1.0 \
    --intermediate_L6_weight 0.5 \
    --num_epochs 300 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 0.1 \
    --curriculum_start 0.3 \
    --curriculum_end 1.0 \
    --curriculum_epochs 100 \
    --save_audio_interval 50 \
    --use_amp \
    2>&1 | tee exp_0112_intermediate/exp_k_v4.log

echo "============================================================"
echo "Exp K v4 完成!"
echo "結果保存於: exp_0112_intermediate/runs/${EXP_NAME}"
echo "============================================================"

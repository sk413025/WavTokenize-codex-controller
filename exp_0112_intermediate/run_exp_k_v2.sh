#!/bin/bash
# ============================================================
# Exp K v2: 修正版中間層監督訓練
# ============================================================
#
# 修正問題:
#   - 原版 intermediate MSE Loss 過大 (L6=1546 vs feature=0.27)
#   - 導致中間層 Loss 主導訓練 (佔比 199.5%)
#
# 修正方案:
#   - 使用 Normalized MSE (除以 feature dimension)
#   - Loss 自動縮放到合理範圍 (~1.0)
#
# 執行:
#   bash exp_0112_intermediate/run_exp_k_v2.sh
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
EXP_NAME="exp_k_v2_${TIMESTAMP}"

echo "============================================================"
echo "Exp K v2: 修正版中間層監督"
echo "============================================================"
echo "實驗名稱: ${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo ""
echo "修正內容:"
echo "  - 使用 Cosine Similarity Loss (1 - cos_sim)"
echo "  - intermediate_weight: 1.0 (Loss 約在 0-1 範圍)"
echo "  - lr: 1e-4 (與 Exp I 一致)"
echo "============================================================"

python exp_0112_intermediate/train_v2.py \
    --exp_name "${EXP_NAME}" \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lr 1e-4 \
    --loss_type cosine \
    --intermediate_weight 1.0 \
    --intermediate_L4_weight 0.5 \
    --intermediate_L8_weight 0.5 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_advance_epochs 30 \
    --num_epochs 150 \
    --batch_size 8 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    2>&1 | tee exp_0112_intermediate/runs/${EXP_NAME}.log

echo "============================================================"
echo "Exp K v2 完成!"
echo "結果保存於: exp_0112_intermediate/runs/${EXP_NAME}"
echo "============================================================"

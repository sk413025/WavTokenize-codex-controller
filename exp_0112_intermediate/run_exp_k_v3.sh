#!/bin/bash
# ============================================================
# Exp K v3: 完整中間層監督訓練 (4 層)
# ============================================================
#
# 配置:
#   L3  (0.5): low_level 代表, Cosine Loss
#   L5  (0.8): mid_level 協同, Cosine Loss
#   L6  (1.0): 噪音處理核心, Cosine Loss
#   L10 (0.3): 語義錨點, MSE Loss (因為本來就穩定)
#
# 基於分析:
#   - exp_1231_feature: L5-L6 是噪音處理核心 (敏感度 0.71-0.79)
#   - 本次分析: L10 是最穩定層 (cos_sim=0.946)
#
# 執行:
#   bash exp_0112_intermediate/run_exp_k_v3.sh
# ============================================================

set -e

# 設定環境
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 實驗名稱與時間戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="exp_k_v3_${TIMESTAMP}"

echo "============================================================"
echo "Exp K v3: 完整中間層監督 (4 層)"
echo "============================================================"
echo "實驗名稱: ${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo ""
echo "監督配置:"
echo "  L3  (weight=0.5): low_level 代表,     Cosine Loss"
echo "  L5  (weight=0.8): mid_level 協同,     Cosine Loss"
echo "  L6  (weight=1.0): 噪音處理核心,       Cosine Loss"
echo "  L10 (weight=0.3): 語義錨點,           MSE Loss"
echo ""
echo "理論依據:"
echo "  - exp_1231_feature: L5-L6 mid-level 敏感度 0.71-0.79 (最高)"
echo "  - 本次分析: L10 cos_sim=0.946 (最穩定)"
echo "============================================================"

python exp_0112_intermediate/train_v3.py \
    --exp_name "${EXP_NAME}" \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lr 1e-4 \
    --intermediate_weight 1.0 \
    --intermediate_L3_weight 0.5 \
    --intermediate_L5_weight 0.8 \
    --intermediate_L6_weight 1.0 \
    --intermediate_L10_weight 0.3 \
    --target_scale 1.0 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_advance_epochs 30 \
    --num_epochs 300 \
    --batch_size 8 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    2>&1 | tee exp_0112_intermediate/runs/${EXP_NAME}.log

echo "============================================================"
echo "Exp K v3 完成!"
echo "結果保存於: exp_0112_intermediate/runs/${EXP_NAME}"
echo "============================================================"

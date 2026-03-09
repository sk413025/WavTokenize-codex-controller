#!/bin/bash
# ============================================================
# Exp L: Multi-Position Adapter 去噪實驗
# ============================================================
# 
# 實驗目的:
#   驗證增加 Adapter 數量和容量能否在保持聽感優勢的同時提升數值性能
#
# 架構設計:
#   - 多位置 Adapter (L2, L4, L6, L8) 覆蓋更多噪音敏感層
#   - 更大的 Bottleneck (input_dim // 2) 增加去噪容量
#   - 預估參數量: ~70K (比 Exp J 的 8K 增加約 8 倍)
#
# 基於 Exp J 經驗:
#   - Exp J 聽感好但數值差 → 容量不足 + 覆蓋不夠
#   - 期望: 數值提升到接近 Exp I 水平，同時保持聽感優勢
#
# 執行:
#   bash exp_0113/run_exp_l.sh
# ============================================================

set -e  # 遇到錯誤就停止

# 設定環境
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 實驗名稱與時間戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="exp_l_multi_adapter_${TIMESTAMP}"

echo "============================================================"
echo "Exp L: Multi-Position Adapter Denoising"
echo "============================================================"
echo "實驗名稱: ${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo "============================================================"

# 執行訓練
python exp_0113/train_exp_l.py \
    --exp_name "${EXP_NAME}" \
    --adapter_positions "1,4,7,10" \
    --reduction_factor 2 \
    --adapter_dropout 0.1 \
    --adapter_init_scale 0.01 \
    --lr 1e-4 \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_increment 0.1 \
    --phase_advance_epochs 30 \
    --weight_decay 0.01 \
    --batch_size 8 \
    --num_epochs 300 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 2 \
    --encoder_stride 320

echo "============================================================"
echo "Exp L 完成!"
echo "結果保存於: exp_0113/runs/${EXP_NAME}"
echo "============================================================"

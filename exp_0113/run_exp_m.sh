#!/bin/bash
# ============================================================
# Exp M: Adapter + LoRA 混合去噪實驗
# ============================================================
# 
# 實驗目的:
#   結合 Adapter 的聽感優勢與 LoRA 的數值性能
#
# 架構設計:
#   - 淺層 (L2, L4): 大 Adapter (reduction=2) → 激進去噪
#   - 中層 (L6): 小 Adapter (reduction=4) + 小 LoRA (rank=32) → 細化去噪
#   - 深層 (L9-L16): 微量 LoRA (rank=16) → 輕微調整
#
# 設計理念:
#   - Adapter: 專責去噪，產生更自然的特徵 (Exp J 發現)
#   - LoRA: 輕微調整深層表示，改善數值性能 (Exp I 證明)
#   - 分層策略: 淺層重去噪、深層輕調整
#
# 預估參數量: ~200K-300K
#   - Adapter: ~70K
#   - LoRA: ~150K-230K
#
# 執行:
#   bash exp_0113/run_exp_m.sh
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
EXP_NAME="exp_m_adapter_lora_${TIMESTAMP}"

echo "============================================================"
echo "Exp M: Adapter + LoRA Hybrid Denoising"
echo "============================================================"
echo "實驗名稱: ${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo "============================================================"

# 執行訓練
python exp_0113/train_exp_m.py \
    --exp_name "${EXP_NAME}" \
    --shallow_adapter_positions "1,4" \
    --shallow_reduction 2 \
    --mid_adapter_positions "7" \
    --mid_reduction 4 \
    --mid_lora_layers "4,7" \
    --mid_lora_rank 32 \
    --deep_lora_layers "10,13,16" \
    --deep_lora_rank 16 \
    --dropout 0.1 \
    --init_scale 0.01 \
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
echo "Exp M 完成!"
echo "結果保存於: exp_0113/runs/${EXP_NAME}"
echo "============================================================"

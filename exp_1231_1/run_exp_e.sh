#!/bin/bash
# =============================================================
# Exp E: 分階段訓練 (Progressive Training)
# =============================================================
#
# 核心概念：
# - 階段 1: 只訓練淺層 (L0-L4)，讓它先學會處理噪音
# - 階段 2: 解凍中層 (L5-L8)，讓噪音敏感層開始學習
# - 階段 3: 解凍深層 (L9-L17)，讓深層適應新的輸入分布
#
# 這樣可以確保：
# 1. 淺層先學會去噪，而不是讓深層硬記
# 2. 深層只需要「適應」而非「學習去噪」
# =============================================================

cd /home/sbplab/ruizi/WavTokenize-feature-analysis/exp_1231_1

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

echo "============================================================"
echo "Exp E: 分階段訓練 (Progressive Training)"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Training Schedule:"
echo "  Phase 1 (L0-L4):   100 epochs - 淺層學習去噪"
echo "  Phase 2 (L0-L8):   100 epochs - 中層加入學習"
echo "  Phase 3 (L0-L17):  100 epochs - 深層適應"
echo "  Total:             300 epochs"
echo "============================================================"

# 創建輸出目錄
mkdir -p runs/exp_e_progressive

python train_progressive.py \
    --exp_name exp_e_progressive \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --phase1_epochs 100 \
    --phase2_epochs 100 \
    --phase3_epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --warmup_epochs 10 \
    --gradient_accumulation_steps 2 \
    --use_amp \
    --seed 42 \
    2>&1 | tee runs/exp_e_progressive/train.log

echo ""
echo "============================================================"
echo "Experiment completed at: $(date)"
echo "============================================================"

#!/bin/bash
# =============================================================
# Exp 1231_1: 凍結深層實驗 (Exp B)
# =============================================================
#
# 基於 exp_1231_feature 分析結論：
# - 噪音主要影響 mid-level (L5-L6)
# - 深層 (L13-L15) 對噪音魯棒但 LoRA 變化最大
# - 這與降噪任務需求相反
#
# 實驗策略：
# - 只對淺/中層 (L0-L8) 加 LoRA
# - 完全凍結深層 (L9-L17)
#
# 預期結果：
# - 如果 Val acc 提升 → 確認深層變化是問題
# - 如果 Val acc 下降 → 需要 Exp A (深層低 rank)
# =============================================================

cd /home/sbplab/ruizi/WavTokenize-feature-analysis/exp_1231_1

# 設定 GPU
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "Exp 1231_1: 凍結深層實驗 (Exp B)"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Layer configuration:"
echo "  - L0-L8 (shallow_mid): LoRA rank=256"
echo "  - L9-L17 (deep): FROZEN"
echo "============================================================"

python train_shallow_lora.py \
    --exp_name exp_b_freeze_deep \
    --lora_layers shallow_mid \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_increment 0.1 \
    --phase_advance_epochs 30 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --num_epochs 300 \
    --warmup_epochs 10 \
    --gradient_accumulation_steps 2 \
    --use_amp \
    --use_scheduler \
    --seed 42 \
    2>&1 | tee runs/exp_b_freeze_deep/train.log

echo ""
echo "============================================================"
echo "Experiment completed at: $(date)"
echo "============================================================"

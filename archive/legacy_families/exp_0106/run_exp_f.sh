#!/bin/bash
# =============================================================
# Exp F: 差異化學習率實驗
# =============================================================
# 淺層 (L0-L8) LR=1e-4，深層 (L9-L17) LR=1e-5
# LR ratio: 10x
# =============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 創建輸出目錄
mkdir -p exp_0106/runs/exp_f_diff_lr

echo "============================================================"
echo "Exp F: Differential Learning Rate"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  - Shallow (L0-L8): LR=1e-4"
echo "  - Deep (L9-L17): LR=1e-5"
echo "  - LR ratio: 10x"
echo "============================================================"

python -u exp_0106/train.py \
    --exp_name exp_f_diff_lr \
    --output_dir /home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0106/runs/exp_f_diff_lr \
    --model_type diff_lr \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lr_shallow 1e-4 \
    --lr_deep 1e-5 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_increment 0.1 \
    --phase_advance_epochs 30 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --num_epochs 300 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 2 \
    2>&1 | tee exp_0106/runs/exp_f_diff_lr/train.log

echo ""
echo "============================================================"
echo "Experiment completed at: $(date)"
echo "============================================================"

#!/bin/bash
# =============================================================
# Exp G: 全層 LoRA + 深層強 L2 正則化
# =============================================================
# 所有層 Rank=256，但對深層施加更強的 L2 正則化 (0.5)
# 與 Exp H (L2=0.1) 比較：Exp G 用更強的正則化
# =============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 創建輸出目錄
mkdir -p exp_0106/runs/exp_g_strong_l2

echo "============================================================"
echo "Exp G: Strong L2 Regularization on Deep Layers"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  - All layers: Rank=256"
echo "  - Deep (L9-L17): L2 weight=0.5"
echo "============================================================"

python -u exp_0106/train.py \
    --exp_name exp_g_strong_l2 \
    --output_dir /home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0106/runs/exp_g_strong_l2 \
    --model_type diff_rank \
    --rank_shallow 256 \
    --rank_deep 32 \
    --alpha_shallow 512 \
    --alpha_deep 64 \
    --lora_dropout 0.2 \
    --lr 1e-4 \
    --l2_reg_weight 0.5 \
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
    2>&1 | tee exp_0106/runs/exp_g_strong_l2/train.log

echo ""
echo "============================================================"
echo "Experiment completed at: $(date)"
echo "============================================================"

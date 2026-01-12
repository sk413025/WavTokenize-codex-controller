#!/bin/bash
# exp_0112: Exp I - 三區差異化訓練 (中層強化)

# 環境設定
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Exp I: Three-Zone Mid-Focus Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Shallow (L0-L4): LR=5e-6, L2=0.01"
echo "  Middle (L5-L8):  LR=2e-5, L2=0  ★ Focus Zone"
echo "  Deep (L9-L17):   LR=1e-5, L2=0"
echo ""
echo "  Seed: 42"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

python exp_0112/train.py \
    --exp_name exp_i_mid_focus \
    --seed 42 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lr_shallow 5e-6 \
    --lr_middle 2e-5 \
    --lr_deep 1e-5 \
    --shallow_l2_weight 0.01 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_increment 0.1 \
    --phase_advance_epochs 30 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --num_epochs 300 \
    --num_workers 4 \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 2 \
    --use_amp

echo ""
echo "Training completed!"
echo "Results saved to: exp_0112/runs/exp_i_mid_focus/"
echo ""
echo "Output files:"
echo "  - training_curves.png"
echo "  - history.json"
echo "  - best_model.pt"
echo "  - audio_samples/train/"
echo "  - audio_samples/val/"

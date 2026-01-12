#!/bin/bash
# exp_0112_adapter: Exp J - 中層 Adapter 去噪

# 環境設定
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Exp J: Mid-Layer Adapter Denoising"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Adapter Position: After L4 (before noise-sensitive L5-L8)"
echo "  Adapter Type: simple (bottleneck)"
echo "  Hidden Dim: input_dim // 4"
echo "  Init Scale: 0.01 (small initial impact)"
echo "  LR: 1e-4"
echo ""
echo "  Only Adapter parameters are trainable!"
echo "  Original WavTokenizer weights are frozen."
echo ""
echo "  Seed: 42"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

python exp_0112_adapter/train.py \
    --exp_name exp_j_adapter \
    --seed 42 \
    --adapter_position 4 \
    --adapter_type simple \
    --adapter_dropout 0.1 \
    --adapter_init_scale 0.01 \
    --lr 1e-4 \
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
echo "Results saved to: exp_0112_adapter/runs/exp_j_adapter/"
echo ""
echo "Output files:"
echo "  - training_curves.png"
echo "  - history.json"
echo "  - best_model.pt"
echo "  - audio_samples/train/"
echo "  - audio_samples/val/"

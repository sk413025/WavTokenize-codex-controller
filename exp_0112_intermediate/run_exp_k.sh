#!/bin/bash
# exp_0112_intermediate: Exp K - 中間層監督訓練

# 環境設定
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 指定 GPU
export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Exp K: Intermediate Layer Supervision"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Supervised layers: L4 (after 1st downsample), L8 (after 2nd downsample)"
echo "  Intermediate weight: 0.5"
echo "  L4 weight: 0.5, L8 weight: 0.5"
echo ""
echo "  Loss = Feature + Triplet + 0.5 × (L4_MSE + L8_MSE)"
echo ""
echo "  Full LoRA on all 18 layers"
echo "  LR: 1e-4"
echo "  Seed: 42"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

python exp_0112_intermediate/train.py \
    --exp_name exp_k_intermediate \
    --seed 42 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lr 1e-4 \
    --intermediate_weight 0.5 \
    --intermediate_L4_weight 0.5 \
    --intermediate_L8_weight 0.5 \
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
echo "Results saved to: exp_0112_intermediate/runs/exp_k_intermediate/"
echo ""
echo "Output files:"
echo "  - training_curves.png"
echo "  - history.json"
echo "  - best_model.pt"
echo "  - audio_samples/train/"
echo "  - audio_samples/val/"

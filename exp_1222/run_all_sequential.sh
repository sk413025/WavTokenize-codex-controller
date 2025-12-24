#!/bin/bash
# 依序運行 Exp58 → Exp57 → Exp56
#
# 順序說明:
# 1. Exp58: 微調 Exp48 (最重要，先跑)
# 2. Exp57: Hybrid Loss (從零開始)
# 3. Exp56: 純 Audio Loss (從零開始)
#
# 使用方式:
#   nohup bash exp_1222/run_all_sequential.sh &
#   或指定 GPU:
#   CUDA_VISIBLE_DEVICES=0 nohup bash exp_1222/run_all_sequential.sh &

set -e  # 遇到錯誤就停止

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Sequential Training: Exp58 → Exp57 → Exp56"
echo "Using GPU: 0"
echo "Start time: $(date)"
echo "=============================================="

# ==================== Exp58 ====================
echo ""
echo "=============================================="
echo "[1/3] Starting Exp58: Finetune Exp48"
echo "Time: $(date)"
echo "=============================================="

python exp_1222/train.py \
    --exp_name exp58_finetune_exp48 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --feature_weight 0.05 \
    --triplet_weight 0.05 \
    --triplet_margin 0.2 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 3 \
    --grad_clip 1.0 \
    --resume exp_1217/runs/exp48_best_config/best_model.pt \
    2>&1 | tee exp_1222/exp58.log

echo ""
echo "[1/3] Exp58 完成! Time: $(date)"
echo ""

# ==================== Exp57 ====================
echo ""
echo "=============================================="
echo "[2/3] Starting Exp57: Hybrid Loss"
echo "Time: $(date)"
echo "=============================================="

python exp_1222/train.py \
    --exp_name exp57_hybrid \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --feature_weight 0.1 \
    --triplet_weight 0.1 \
    --triplet_margin 0.2 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 5 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1222/exp57.log

echo ""
echo "[2/3] Exp57 完成! Time: $(date)"
echo ""

# ==================== Exp56 ====================
echo ""
echo "=============================================="
echo "[3/3] Starting Exp56: Pure Audio Loss"
echo "Time: $(date)"
echo "=============================================="

python exp_1222/train.py \
    --exp_name exp56_audio_loss \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --feature_weight 0.0 \
    --triplet_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 5 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1222/exp56.log

echo ""
echo "[3/3] Exp56 完成! Time: $(date)"
echo ""

# ==================== 完成 ====================
echo "=============================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Exp58: exp_1222/runs/exp58_finetune_exp48/"
echo "  - Exp57: exp_1222/runs/exp57_hybrid/"
echo "  - Exp56: exp_1222/runs/exp56_audio_loss/"

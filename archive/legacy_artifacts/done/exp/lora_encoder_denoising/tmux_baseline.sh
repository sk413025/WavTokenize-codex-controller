#!/bin/bash

# LoRA Encoder Denoising - Baseline Training (tmux)
# Phase 1: Establish performance baseline

cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising

# 實驗參數
EXP_NAME="baseline_r16_lr5e5"
TMUX_SESSION="lora_${EXP_NAME}"
GPU_ID=1  # 使用 RTX 2080 Ti (GPU 1)
EPOCHS=50
BATCH_SIZE=8
LR=5e-5
LORA_RANK=16
LORA_ALPHA=32
NUM_WORKERS=4
SEED=42

# Loss weights
FEATURE_LOSS_WEIGHT=1.0
DISTANCE_LOSS_WEIGHT=0.1
VQ_LOSS_WEIGHT=0.01

# 創建 experiments 目錄
mkdir -p experiments

echo "========================================"
echo "  LoRA Encoder Denoising Training"
echo "  使用 tmux session: $TMUX_SESSION"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "GPU: $GPU_ID (RTX 2080 Ti)"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
echo "========================================"
echo ""

# 檢查 tmux session 是否已存在
if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
    echo "⚠️  Tmux session '$TMUX_SESSION' 已存在"
    echo "選項:"
    echo "  1. tmux attach -t $TMUX_SESSION  (連接到現有 session)"
    echo "  2. tmux kill-session -t $TMUX_SESSION  (刪除後重新執行)"
    exit 1
fi

# 創建 tmux session 並啟動訓練
tmux new-session -d -s $TMUX_SESSION

# 在 tmux session 中執行訓練
tmux send-keys -t $TMUX_SESSION "cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising" C-m
tmux send-keys -t $TMUX_SESSION "CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \\
  --exp_name $EXP_NAME \\
  --num_epochs $EPOCHS \\
  --batch_size $BATCH_SIZE \\
  --learning_rate $LR \\
  --lora_rank $LORA_RANK \\
  --lora_alpha $LORA_ALPHA \\
  --feature_loss_weight $FEATURE_LOSS_WEIGHT \\
  --distance_loss_weight $DISTANCE_LOSS_WEIGHT \\
  --vq_loss_weight $VQ_LOSS_WEIGHT \\
  --num_workers $NUM_WORKERS \\
  --seed $SEED \\
  2>&1 | tee experiments/${EXP_NAME}.log" C-m

echo "✓ 訓練已在 tmux session 中啟動"
echo ""
echo "Tmux Session: $TMUX_SESSION"
echo "Log File: experiments/${EXP_NAME}.log"
echo ""
echo "常用指令:"
echo "  tmux attach -t $TMUX_SESSION        # 連接到訓練 session"
echo "  tmux detach                         # 斷開 session (訓練繼續)"
echo "  tmux kill-session -t $TMUX_SESSION  # 停止訓練並關閉 session"
echo ""
echo "監控訓練進度:"
echo "  tail -f experiments/${EXP_NAME}.log"
echo "  tensorboard --logdir experiments/ --port 6006"
echo "  watch -n 2 nvidia-smi  # 監控 GPU 使用"
echo ""
echo "預估訓練時間: ~5-6 小時 (50 epochs)"
echo ""

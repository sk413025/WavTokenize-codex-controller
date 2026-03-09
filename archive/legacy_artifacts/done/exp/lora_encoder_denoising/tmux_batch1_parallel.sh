#!/bin/bash

# LoRA Encoder Denoising - Batch 1 Parallel Training (tmux)
# 同時執行: Baseline (rank=16) + Small LoRA (rank=8)
# 使用 GPU 1 + GPU 2 (2x RTX 2080 Ti)

cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising

# 共同參數
EPOCHS=50
BATCH_SIZE=8
LR=5e-5
NUM_WORKERS=4
SEED=42
FEATURE_LOSS_WEIGHT=1.0
DISTANCE_LOSS_WEIGHT=0.1
VQ_LOSS_WEIGHT=0.01

# Experiment 1: Baseline (rank=16)
EXP_NAME_1="baseline_r16_lr5e5"
TMUX_SESSION_1="lora_${EXP_NAME_1}"
GPU_ID_1=1
LORA_RANK_1=16
LORA_ALPHA_1=32

# Experiment 2: Small LoRA (rank=8)
EXP_NAME_2="lora_r8_lr5e5"
TMUX_SESSION_2="lora_${EXP_NAME_2}"
GPU_ID_2=2
LORA_RANK_2=8
LORA_ALPHA_2=16

# 創建 experiments 目錄
mkdir -p experiments

echo "========================================"
echo "  Batch 1: Parallel Training (tmux)"
echo "  使用 GPU 1 + GPU 2"
echo "========================================"
echo ""

# 檢查 tmux sessions 是否已存在
for session in $TMUX_SESSION_1 $TMUX_SESSION_2; do
    if tmux has-session -t $session 2>/dev/null; then
        echo "⚠️  Tmux session '$session' 已存在"
        echo "請先執行: tmux kill-session -t $session"
        exit 1
    fi
done

# Experiment 1: Baseline (rank=16) on GPU 1
echo "啟動 Experiment 1: $EXP_NAME_1"
echo "  Tmux Session: $TMUX_SESSION_1"
echo "  GPU: $GPU_ID_1 (RTX 2080 Ti)"
echo "  LoRA Rank: $LORA_RANK_1"

tmux new-session -d -s $TMUX_SESSION_1
tmux send-keys -t $TMUX_SESSION_1 "cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising" C-m
tmux send-keys -t $TMUX_SESSION_1 "CUDA_VISIBLE_DEVICES=$GPU_ID_1 python train.py \\
  --exp_name $EXP_NAME_1 \\
  --num_epochs $EPOCHS \\
  --batch_size $BATCH_SIZE \\
  --learning_rate $LR \\
  --lora_rank $LORA_RANK_1 \\
  --lora_alpha $LORA_ALPHA_1 \\
  --feature_loss_weight $FEATURE_LOSS_WEIGHT \\
  --distance_loss_weight $DISTANCE_LOSS_WEIGHT \\
  --vq_loss_weight $VQ_LOSS_WEIGHT \\
  --num_workers $NUM_WORKERS \\
  --seed $SEED \\
  2>&1 | tee experiments/${EXP_NAME_1}.log" C-m

echo "  ✓ 已啟動"
echo ""
sleep 2

# Experiment 2: Small LoRA (rank=8) on GPU 2
echo "啟動 Experiment 2: $EXP_NAME_2"
echo "  Tmux Session: $TMUX_SESSION_2"
echo "  GPU: $GPU_ID_2 (RTX 2080 Ti)"
echo "  LoRA Rank: $LORA_RANK_2"

tmux new-session -d -s $TMUX_SESSION_2
tmux send-keys -t $TMUX_SESSION_2 "cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising" C-m
tmux send-keys -t $TMUX_SESSION_2 "CUDA_VISIBLE_DEVICES=$GPU_ID_2 python train.py \\
  --exp_name $EXP_NAME_2 \\
  --num_epochs $EPOCHS \\
  --batch_size $BATCH_SIZE \\
  --learning_rate $LR \\
  --lora_rank $LORA_RANK_2 \\
  --lora_alpha $LORA_ALPHA_2 \\
  --feature_loss_weight $FEATURE_LOSS_WEIGHT \\
  --distance_loss_weight $DISTANCE_LOSS_WEIGHT \\
  --vq_loss_weight $VQ_LOSS_WEIGHT \\
  --num_workers $NUM_WORKERS \\
  --seed $SEED \\
  2>&1 | tee experiments/${EXP_NAME_2}.log" C-m

echo "  ✓ 已啟動"
echo ""

echo "========================================"
echo "  Batch 1 訓練已啟動 (2 個實驗)"
echo "========================================"
echo ""
echo "Experiment 1 (Baseline, rank=16):"
echo "  Tmux Session: $TMUX_SESSION_1"
echo "  GPU: $GPU_ID_1"
echo "  Log: experiments/${EXP_NAME_1}.log"
echo ""
echo "Experiment 2 (Small LoRA, rank=8):"
echo "  Tmux Session: $TMUX_SESSION_2"
echo "  GPU: $GPU_ID_2"
echo "  Log: experiments/${EXP_NAME_2}.log"
echo ""
echo "預估訓練時間: ~5-6 小時 (50 epochs)"
echo ""
echo "查看所有 tmux sessions:"
echo "  tmux ls"
echo ""
echo "連接到特定 session:"
echo "  tmux attach -t $TMUX_SESSION_1  # Baseline"
echo "  tmux attach -t $TMUX_SESSION_2  # Small LoRA"
echo ""
echo "監控訓練進度:"
echo "  tail -f experiments/${EXP_NAME_1}.log"
echo "  tail -f experiments/${EXP_NAME_2}.log"
echo "  tensorboard --logdir experiments/ --port 6006"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "停止特定訓練:"
echo "  tmux kill-session -t $TMUX_SESSION_1"
echo "  tmux kill-session -t $TMUX_SESSION_2"
echo ""
echo "停止所有訓練:"
echo "  tmux kill-session -t $TMUX_SESSION_1 && tmux kill-session -t $TMUX_SESSION_2"
echo ""

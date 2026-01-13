#!/bin/bash
# exp_test: Rank 1024 單獨執行

set -e

# 設定環境
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

echo "=========================================="
echo "exp_test: Rank 1024"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo "=========================================="

python exp_test/train.py \
    --exp_name shallow_r1024 \
    --lora_rank 1024 \
    --lora_alpha 2048 \
    --lr 1e-4 \
    --num_epochs 150 \
    --batch_size 8 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_advance_epochs 30 \
    2>&1 | tee exp_test/runs/shallow_r1024.log

echo "=========================================="
echo "Rank 1024 完成!"
echo "=========================================="

#!/bin/bash
# exp_test: Rank 256 單獨執行

set -e

# 設定環境
export CUDA_VISIBLE_DEVICES=1S
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

echo "=========================================="
echo "exp_test: Rank 256"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo "=========================================="

python exp_test/train.py \
    --exp_name shallow_r256 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lr 1e-4 \
    --num_epochs 150 \
    --batch_size 8 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_advance_epochs 30 \
    2>&1 | tee exp_test/runs/shallow_r256.log

echo "=========================================="
echo "Rank 256 完成!"
echo "=========================================="

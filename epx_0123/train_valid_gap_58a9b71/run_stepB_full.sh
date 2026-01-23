#!/usr/bin/env bash
set -e
cd /home/sbplab/ruizi/WavTokenize-feature-analysis
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test
export CUDA_VISIBLE_DEVICES=1

stdbuf -oL -eL python exp_0112_intermediate/analysis/train_valid_gap_58a9b71/stepB_offline_eval.py \
  --device cuda:0 \
  --batch_size 4 \
  --num_workers 0 \
  --progress_every 200

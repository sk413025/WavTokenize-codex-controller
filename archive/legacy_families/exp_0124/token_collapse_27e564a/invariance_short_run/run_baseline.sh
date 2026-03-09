#!/usr/bin/env bash
set -eo pipefail

source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

date
nvidia-smi -L

python exp_0124/token_collapse_27e564a/invariance_short_run/run_invariance_short.py \
  --lambdas 0.0 \
  --max_steps 800 \
  --max_train_samples 2000 \
  --max_val_samples 500 \
  --batch_size 1 \
  --num_workers 0 \
  --use_amp \
  --gradient_accumulation_steps 2 \
  --log_every 50

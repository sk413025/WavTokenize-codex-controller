#!/usr/bin/env bash
set -eo pipefail

source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

echo "CONDA_OK"
which python

export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

date
nvidia-smi -L

set +e
python exp_0124/token_collapse_27e564a/invariance_short_run/run_invariance_short.py \
  --output_root exp_0124/token_collapse_27e564a/invariance_short_run_shift/runs \
  --lambdas 0.05,0.10 \
  --global_shift_k 3 \
  --max_steps 800 \
  --max_train_samples 2000 \
  --max_val_samples 500 \
  --batch_size 1 \
  --num_workers 0 \
  --use_amp \
  --gradient_accumulation_steps 2 \
  --log_every 50

exit_code=$?
echo "EXIT_CODE:${exit_code}"
set -e

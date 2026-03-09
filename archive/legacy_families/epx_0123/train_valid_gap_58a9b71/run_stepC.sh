#!/usr/bin/env bash
set -e
cd /home/sbplab/ruizi/WavTokenize-feature-analysis
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test
export CUDA_VISIBLE_DEVICES=0

stdbuf -oL -eL python exp_0112_intermediate/analysis/train_valid_gap_58a9b71/stepC_data_difficulty_alignment.py

#!/bin/bash
# exp_0128 Phase 2: Codebook Refresh - interval=100, threshold=10 (實驗 4a)

CUDA_VISIBLE_DEVICES=0 python exp_0128/phase2/codebook_refresh/train_codebook_refresh.py \
    --refresh_interval 100 \
    --usage_threshold 10 \
    --steps 1000 \
    --batch_size 2 \
    --grad_accum 2 \
    --lr 1e-4 \
    --output_dir exp_0128/phase2/codebook_refresh/exp4a_interval_100_thresh_10 \
    --seed 42 \
    --device cuda:0 \
    --eval_interval 200 \
    --save_checkpoint_every 200 \
    --save_audio_interval 500

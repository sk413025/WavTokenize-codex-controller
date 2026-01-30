#!/bin/bash
# exp_0128 Phase 2: Codebook Refresh - interval=50, threshold=5 (實驗 4b)

CUDA_VISIBLE_DEVICES=1 python exp_0128/phase2/codebook_refresh/train_codebook_refresh.py \
    --refresh_interval 50 \
    --usage_threshold 5 \
    --steps 1000 \
    --batch_size 2 \
    --grad_accum 2 \
    --lr 1e-4 \
    --output_dir exp_0128/phase2/codebook_refresh/exp4b_interval_50_thresh_5 \
    --seed 42 \
    --device cuda:1 \
    --eval_interval 200 \
    --save_checkpoint_every 200 \
    --save_audio_interval 500

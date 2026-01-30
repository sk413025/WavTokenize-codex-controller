#!/bin/bash
# exp_0128 Phase 2: Entropy Regularization - λ=0.05 (實驗 3b)

CUDA_VISIBLE_DEVICES=1 python exp_0128/phase2/entropy_regularization/train_entropy_reg.py \
    --lambda_entropy 0.05 \
    --steps 1000 \
    --batch_size 2 \
    --grad_accum 2 \
    --lr 1e-4 \
    --output_dir exp_0128/phase2/entropy_regularization/exp3b_lambda_0.05 \
    --seed 42 \
    --device cuda:1 \
    --eval_interval 200 \
    --save_checkpoint_every 200 \
    --save_audio_interval 500

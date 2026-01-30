#!/bin/bash
# exp_0128 Phase 2: Entropy Regularization - λ=0.1 (實驗 3c)

CUDA_VISIBLE_DEVICES=0 python exp_0128/phase2/entropy_regularization/train_entropy_reg.py \
    --lambda_entropy 0.1 \
    --steps 1000 \
    --batch_size 2 \
    --grad_accum 2 \
    --lr 1e-4 \
    --output_dir exp_0128/phase2/entropy_regularization/exp3c_lambda_0.1 \
    --seed 42 \
    --device cuda:0 \
    --eval_interval 200 \
    --save_checkpoint_every 200 \
    --save_audio_interval 500

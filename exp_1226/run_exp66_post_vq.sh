#!/bin/bash
# Exp66: Post-VQ Feature Loss 實驗
#
# 核心概念：
# - 診斷發現：Pre-VQ Cosine Sim = 0.495，Post-VQ Cosine Sim = 0.9325
# - 目前訓練只優化 Pre-VQ encoder output
# - 直接優化 Post-VQ quantized features 應該能更有效改善音質
#
# 新增 Loss:
# - Post-VQ Feature Loss: MSE(VQ(student), VQ(teacher))
# - Post-VQ Cosine Loss: 最大化 Post-VQ cosine similarity
# - 使用 Straight-Through Estimator 讓梯度穿過 VQ

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1226/train_exp66_post_vq.py \
    --exp_name exp66_post_vq \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1226/runs/exp66_post_vq \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --post_vq_feature_weight 0.5 \
    --post_vq_cosine_weight 0.5 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 300 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1226/exp66.log

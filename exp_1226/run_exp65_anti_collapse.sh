#!/bin/bash
# Exp65: Anti-Collapse + Frame-Tolerant 實驗
#
# 問題診斷：
# - Student encoder 發生 mode collapse
# - Token accuracy 只有 ~0.9%
# - Student codes 集中在少數幾個 (1760, 1834, 1623...)
# - 73.3% 樣本有超過半個 frame 的時間偏移
#
# 解決方案：
# - Code Entropy Loss: 鼓勵 code distribution 更均勻
# - Feature Diversity Loss: 懲罰 batch 內 features 太相似
# - Batch Contrastive Loss: 確保不同輸入產生不同輸出
# - Frame-Tolerant Loss: 允許 ±1 frame 容忍度處理時間偏移

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1226/train_exp65_anti_collapse.py \
    --exp_name exp65_anti_collapse \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1226/runs/exp65_anti_collapse \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --entropy_weight 0.1 \
    --diversity_weight 0.1 \
    --contrastive_weight 0.1 \
    --diversity_margin 0.5 \
    --contrastive_temperature 0.1 \
    --use_frame_tolerant \
    --frame_tolerance 1 \
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
    2>&1 | tee exp_1226/exp65.log

#!/bin/bash
# Exp55: Gradient Accumulation 實驗
#
# 目的: 驗證高 LoRA rank 是否因 batch size 過小而無法發揮效果
#
# 配置:
# - lora_rank: 256 (高容量)
# - lora_alpha: 512
# - batch_size: 8 (較小，避免 OOM)
# - gradient_accumulation_steps: 2
# - 等效 batch_size: 8 × 2 = 16
#
# 對照:
# - Exp52: rank=256, batch_size=10, 無 accumulation → 等效 batch=10
# - Exp55: rank=256, batch_size=8,  accumulation=2  → 等效 batch=16
#
# 預期:
# - 如果 Exp55 > Exp52，證明 batch size 是限制因素
# - 高 rank 確實有用，只是需要更穩定的梯度

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1219/train.py \
    --exp_name exp55_grad_accum \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_epochs 200 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1219/exp55.log

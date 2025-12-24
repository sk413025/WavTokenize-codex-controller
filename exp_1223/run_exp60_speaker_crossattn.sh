#!/bin/bash
# Exp60-b: Speaker-Conditioned Training (Cross-Attention)
#
# 與 Exp60 (FiLM) 的對照實驗
# 使用 Cross-Attention + Gate 方式進行 speaker conditioning
# (參考 c_code/exp3-1 的設計)
#
# 配置:
# - 基礎: rank=256, alpha=512 (同 Exp55)
# - Speaker: Cross-Attention conditioning
#   - Token features (Q) attend to Speaker embedding (K, V)
#   - Learnable gate: α * features + (1-α) * attn_output
# - LR: 5e-5 (同 Exp59)
# - Grad Accum: 4 (同 Exp59)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1223/train_speaker.py \
    --exp_name exp60b_speaker_crossattn \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --speaker_condition_type cross_attention \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 300 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 15 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1223/exp60b.log

#!/bin/bash
# Exp61: Speaker-Weighted Loss
#
# 核心概念:
# - 不改變 features，只在 loss 層面加入 speaker information
# - 根據 speaker embedding 與 training speakers 的相似度調整 loss weight
# - 對 unseen speakers 給予較低的 penalty (min_weight=0.5)
#
# 相比 Exp60 (FiLM):
# - Exp60: 改變 features → 破壞 VQ mapping → 音質變差
# - Exp61: 只調整 loss weight → 保持 VQ mapping → 音質不受影響
#
# 配置:
# - 使用原始 LoRA 模型 (不用 speaker conditioning module)
# - Loss: Feature + Triplet (與 Exp59 相同)
# - Speaker weighting: min_weight=0.5, temperature=1.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1223/train_speaker_weighted.py \
    --exp_name exp61_speaker_weighted \
    --output_dir /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1223/runs/exp61_speaker_weighted \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --speaker_min_weight 0.5 \
    --speaker_temperature 1.0 \
    --use_speaker_weighting \
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
    2>&1 | tee exp_1223/exp61.log

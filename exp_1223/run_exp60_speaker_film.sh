#!/bin/bash
# Exp60: Speaker-Conditioned Training (FiLM)
#
# 設計思路:
# 1. 基於 Exp55/Exp59 的 LoRA fine-tuning
# 2. 加入 Speaker Conditioning (FiLM method)
#    - FiLM: Feature-wise Linear Modulation
#    - y = γ * x + β, 其中 γ, β 由 speaker embedding 生成
# 3. 參考 c_code/exp3-1 的 Cross-Attention + Gate 設計
#
# 配置:
# - 基礎: rank=256, alpha=512 (同 Exp55)
# - Speaker: FiLM conditioning (256 → 512 → γ, β)
# - LR: 5e-5 (同 Exp59)
# - Grad Accum: 4 (同 Exp59)
#
# 預期:
# - Speaker conditioning 可能幫助模型更好地適應不同說話者的 token 分佈
# - Val speakers (boy7, girl9, boy8) 與 Train speakers 不重疊，是 zero-shot 測試

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1223/train_speaker.py \
    --exp_name exp60_speaker_film \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --speaker_condition_type film \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 200 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 15 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1223/exp60.log

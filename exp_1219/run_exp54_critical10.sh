#!/bin/bash
# Exp54: 關鍵 10 層實驗
#
# 目的: 測試減少層數但覆蓋關鍵語義層的效果
#
# 層配置 (critical_10):
# - model.0: 輸入投影
# - model.3: Downsample 1
# - model.6: Downsample 2
# - model.7.*: 語義提取 ResBlock ★★★ (2 層)
# - model.9: Downsample 3
# - model.10.*: 高階抽象 ResBlock ★★★ (2 層)
# - model.12: Downsample 4
# - model.15: 輸出投影
#
# 與 critical_8 的差異:
# - critical_8 遺漏了 model.7 和 model.10 的語義層
# - critical_10 完整覆蓋這兩個關鍵 ResBlock
#
# 預期:
# - 比 all_18 更低的過擬合風險 (fewer parameters)
# - 比 critical_8 更好的 accuracy (覆蓋語義層)
# - 可能達到 all_18 類似的 accuracy 但更好的 generalization

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1219/train.py \
    --exp_name exp54_critical10 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers critical_10 \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 16 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1219/exp54.log

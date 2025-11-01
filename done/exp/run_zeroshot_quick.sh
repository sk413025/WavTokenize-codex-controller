#!/bin/bash

# Zero-Shot Speaker Denoising - 快速驗證實驗
# 目的: 驗證 speaker conditioning 是否有效
# 配置: 2 speakers (boy1 訓練, girl9 驗證), 10 sentences/speaker, 5 epochs

echo "========================================="
echo "Zero-Shot 快速驗證實驗"
echo "========================================="
echo "配置:"
echo "  - Speakers: boy1 (train), girl9 (val)"
echo "  - Sentences/speaker: 10"
echo "  - Materials: box, papercup, plastic"
echo "  - Epochs: 10"
echo "  - Batch size: 12 (接近 baseline 的 14)"
echo "  - 預計時間: 45-60 分鐘"
echo "========================================="
echo "注意: Batch size=12 是參考 baseline=14 的經驗"
echo "      避免 batch size 過小導致訓練不穩定"
echo "========================================="

python train_zeroshot_quick.py \
    --input_dirs ../../data/raw/box ../../data/raw/papercup ../../data/raw/plastic ../../data/clean/box2\
    --target_dir ../../data/clean/box2 \
    --output_dir ./results/zeroshot_quick_$(date +%Y%m%d_%H%M%S) \
    --num_epochs 100 \
    --batch_size 12 \
    --max_sentences_per_speaker 10 \
    --speaker_encoder ecapa \
    --speaker_dim 256 \
    --learning_rate 1e-4

echo ""
echo "========================================="
echo "實驗完成！"
echo "========================================="
echo "請查看:"
echo "  - 訓練日誌: results/zeroshot_quick_*/training.log"
echo "  - 損失曲線: results/zeroshot_quick_*/loss_curves.png"
echo "  - 最佳模型: results/zeroshot_quick_*/best_model.pth"
echo ""
echo "關鍵指標:"
echo "  - 查看最終 Val Acc"
echo "  - 如果 > 45%: ✅ 架構有效"
echo "  - 如果 40-45%: ⚠️  勉強及格"
echo "  - 如果 < 40%: ❌ 需要調整"

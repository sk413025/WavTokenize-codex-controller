#!/bin/bash

# Zero-Shot Speaker Denoising - 完整實驗
# 目的: 公平對比 baseline
# 配置: 18 speakers (14 train, 4 val), 288 sentences/speaker, 100 epochs

echo "========================================="
echo "Zero-Shot 完整實驗"
echo "========================================="
echo "配置:"
echo "  - Train speakers: 14 人 (與 baseline 相同)"
echo "  - Val speakers: 4 人 (girl9, girl10, boy7, boy8)"
echo "  - Sentences/speaker: 288 (與 baseline 相同)"
echo "  - Materials: box, papercup, plastic"
echo "  - Epochs: 100"
echo "  - Batch size: 14 (與 baseline 相同)"
echo "  - 預計時間: 6-12 小時"
echo "========================================="
echo ""
echo "⚠️  注意："
echo "  - 此實驗將運行較長時間"
echo "  - 確保有足夠的 GPU 記憶體"
echo "  - 建議使用 nohup 在背景運行"
echo "========================================="

python train_zeroshot_full.py \
    --input_dirs ../../data/raw/box ../../data/raw/papercup ../../data/raw/plastic ../../data/clean/box2 \
    --target_dir ../../data/clean/box2 \
    --output_dir ./results/zeroshot_full_$(date +%Y%m%d_%H%M%S) \
    --num_epochs 100 \
    --batch_size 14 \
    --max_sentences_per_speaker 288 \
    --speaker_encoder ecapa \
    --speaker_dim 256 \
    --learning_rate 1e-4

echo ""
echo "========================================="
echo "實驗完成！"
echo "========================================="
echo "請查看:"
echo "  - 訓練日誌: results/zeroshot_full_*/training.log"
echo "  - 損失曲線: results/zeroshot_full_*/loss_curves.png"
echo "  - 音檔樣本: results/zeroshot_full_*/audio_samples/"
echo "  - 最佳模型: results/zeroshot_full_*/best_model.pth"
echo ""
echo "關鍵指標:"
echo "  - 對比 Baseline Val Acc: 38%"
echo "  - 如果 > 45%: ✅ Zero-shot 架構有效"
echo "  - 如果 < 38%: ❌ 不如 baseline"

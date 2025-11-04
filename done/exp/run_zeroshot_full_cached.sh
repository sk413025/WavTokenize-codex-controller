#!/bin/bash

# Zero-Shot Speaker Denoising - 完整實驗（使用緩存版本）
# 目的: 使用預處理緩存，大幅提升訓練速度
# 優勢:
#   - 訓練速度提升 8x (3.2s/batch → 0.4s/batch)
#   - GPU 利用率: 75-90% (從 22-52%)
#   - 100 epochs: 15 小時 (從 115 小時)

echo "========================================="
echo "Zero-Shot 完整實驗（緩存加速版）"
echo "========================================="
echo "配置:"
echo "  - Train speakers: 14 人"
echo "  - Val speakers: 4 人 (girl9, girl10, boy7, boy8)"
echo "  - Sentences/speaker: 288"
echo "  - Epochs: 100"
echo "  - Batch size: 28 (提升 2x，從 14 → 28)"
echo "  - Num workers: 4 (啟用多進程)"
echo "  - 預計時間: 15 小時 (從 115 小時提升 8x)"
echo "========================================="
echo ""
echo "⚠️  前置條件："
echo "  - 確保已運行 run_preprocess.sh 生成緩存"
echo "  - 確保 ./data/train_cache.pt 和 val_cache.pt 存在"
echo "========================================="

# 檢查緩存是否存在
if [ ! -f "./data/train_cache.pt" ]; then
    echo ""
    echo "❌ 錯誤: 訓練集緩存不存在"
    echo "請先運行: bash run_preprocess.sh"
    exit 1
fi

if [ ! -f "./data/val_cache.pt" ]; then
    echo ""
    echo "❌ 錯誤: 驗證集緩存不存在"
    echo "請先運行: bash run_preprocess.sh"
    exit 1
fi

echo ""
echo "✓ 緩存檢查通過"
echo ""

python train_zeroshot_full_cached_analysis.py \
    --cache_dir ./data \
    --output_dir ./results/zeroshot_full_cached_$(date +%Y%m%d_%H%M%S) \
    --num_epochs 100 \
    --batch_size 28 \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --analyze_speakers \
    --speaker_analysis_freq 50

echo ""
echo "========================================="
echo "實驗完成！"
echo "========================================="
echo "請查看:"
echo "  - 訓練日誌: results/zeroshot_full_cached_*/training.log"
echo "  - 損失曲線: results/zeroshot_full_cached_*/loss_curves.png"
echo "  - 音檔樣本: results/zeroshot_full_cached_*/audio_samples/"
echo "  - 最佳模型: results/zeroshot_full_cached_*/best_model.pth"
echo ""
echo "關鍵指標:"
echo "  - 對比 Baseline Val Acc: 38.19%"
echo "  - 如果 > 45%: ✅ Zero-shot 架構有效"
echo "  - 如果 < 38.19%: ❌ 不如 baseline"
echo ""
echo "性能提升:"
echo "  - 訓練速度: 8x 更快 (0.4s/batch vs 3.2s/batch)"
echo "  - GPU 利用率: 75-90% (vs 22-52%)"
echo "  - 總時間: 15 小時 (vs 115 小時)"

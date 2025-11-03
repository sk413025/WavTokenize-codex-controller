#!/bin/bash

# ============================================================================
# 最小化測試：14 位語者，每位 1 句話，訓練 10 epochs
# 用於快速驗證所有組件是否正常工作
# ============================================================================

echo "=========================================="
echo "EXP2 最小化測試"
echo "=========================================="
echo "配置："
echo "  - 語者數: 14 位 (全部)"
echo "  - 每位語者句子數: 1 句"
echo "  - 訓練 epochs: 10"
echo "  - Batch size: 4"
echo "  - Lambda (speaker loss): 0.5"
echo "=========================================="
echo ""

cd /home/sbplab/ruizi/c_code

# 檢查數據是否存在
echo "1. 檢查數據路徑..."
if [ ! -d "data/raw/box" ]; then
    echo "❌ 數據路徑不存在"
    exit 1
fi
echo "✓ 數據路徑正確"
echo ""

# 運行最小化訓練
echo "2. 開始最小化訓練..."
echo "-----------------------------------"

python done/exp2/train_with_speaker.py \
    --input_dirs data/raw/box data/raw/papercup data/raw/plastic \
    --target_dir data/clean/box2 \
    --output_dir ./results/exp2/test_minimal \
    --lambda_speaker 0.5 \
    --speaker_model_type ecapa \
    --num_epochs 100 \
    --batch_size 14 \
    --max_sentences_per_speaker 1 \
    --learning_rate 1e-4 \
    --num_layers 2

echo ""
echo "=========================================="
echo "測試完成！"
echo "=========================================="
echo ""
echo "檢查結果："
echo "  1. 查看日誌: cat ./results/exp2/test_minimal/training.log"
echo "  2. 查看配置: cat ./results/exp2/test_minimal/config.json"
echo "  3. 聆聽樣本: ls ./results/exp2/test_minimal/audio_samples/"
echo ""

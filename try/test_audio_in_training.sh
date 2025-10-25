#!/bin/bash
# 測試訓練中的音頻和頻譜圖儲存功能

set -e

echo "======================================"
echo "測試訓練中的音頻/頻譜圖儲存 (1 epoch)"
echo "======================================"

# 選擇 GPU
export CUDA_VISIBLE_DEVICES=0

# 臨時輸出目錄
TEST_DIR="../results/test_audio_save_training_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "測試配置:"
echo "  - Epochs: 1"
echo "  - 輸出: $TEST_DIR"
echo "  - GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# 運行 1 epoch 訓練
python train_token_denoising_hybrid.py \
    --input_dirs ../data/raw/box \
    --target_dir ../data/clean/box2 \
    --output_dir "$TEST_DIR" \
    --num_epochs 1 \
    --batch_size 4 \
    --num_layers 4 \
    --dropout 0.2 \
    --weight_decay 0.05 \
    --wavtokenizer_config ../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --wavtokenizer_checkpoint ../models/wavtokenizer_large_speech_320_24k.ckpt

echo ""
echo "======================================"
echo "檢查輸出"
echo "======================================"

# 檢查音頻樣本（epoch 1 會在最後保存）
if [ -d "$TEST_DIR/audio_samples" ]; then
    echo "✓ 音頻樣本目錄存在"
    ls -lh "$TEST_DIR/audio_samples"/*/*wav 2>/dev/null | head -5
else
    echo "✗ 音頻樣本目錄不存在"
fi

# 檢查頻譜圖
if [ -d "$TEST_DIR/spectrograms" ]; then
    echo "✓ 頻譜圖目錄存在"  
    ls -lh "$TEST_DIR/spectrograms"/*/*png 2>/dev/null | head -5
else
    echo "✗ 頻譜圖目錄不存在"
fi

echo ""
echo "======================================"
echo "✅ 測試完成"
echo "======================================"
echo "輸出目錄: $TEST_DIR"

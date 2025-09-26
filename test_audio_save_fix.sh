#!/bin/bash
# 音頻樣本保存修復驗證實驗 - 實驗編號: EXP_AUDIO_SAVE_FIX_TEST_20250925
# 目的: 快速驗證修復後的音頻樣本保存功能是否正常工作

# 設定實驗環境
export CUDA_VISIBLE_DEVICES=2
cd /home/sbplab/ruizi/c_code

echo "🔧 開始音頻樣本保存修復驗證實驗..."
echo "📅 實驗時間: $(date)"
echo "🔢 實驗編號: EXP_AUDIO_SAVE_FIX_TEST_20250925"
echo "🎯 目的: 驗證修復後的音頻樣本保存功能"
echo "🖥️  GPU: $CUDA_VISIBLE_DEVICES"
echo "📂 工作目錄: $(pwd)"
echo ""

# 創建實驗輸出目錄
EXPERIMENT_DIR="results/audio_save_fix_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXPERIMENT_DIR

# 運行短期實驗 (只訓練2個epoch來測試音頻保存)
echo "🚀 開始訓練 (僅2個epoch用於測試音頻保存)..."
python wavtokenizer_transformer_denoising.py \
    --config config/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --model_path models/wavtokenizer_large_speech_320_24k.ckpt \
    --num_epochs 2 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --d_model 128 \
    --nhead 2 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --dim_feedforward 256 \
    --max_length 256 \
    --dropout 0.1 \
    --save_every 1 \
    2>&1 | tee logs/audio_save_fix_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "📊 實驗結果檢查:"

# 檢查是否成功生成了音頻樣本
if [ -d "$EXPERIMENT_DIR/audio_samples" ]; then
    echo "✅ 音頻樣本目錄已創建: $EXPERIMENT_DIR/audio_samples"
    
    # 統計音頻文件數量
    AUDIO_COUNT=$(find $EXPERIMENT_DIR/audio_samples -name "*.wav" | wc -l)
    echo "🔊 音頻文件總數: $AUDIO_COUNT"
    
    # 統計頻譜圖數量
    SPEC_COUNT=$(find $EXPERIMENT_DIR/audio_samples -name "*_spec.png" | wc -l)
    echo "📊 頻譜圖總數: $SPEC_COUNT"
    
    # 列出所有生成的文件
    echo "📁 生成的文件列表:"
    ls -la $EXPERIMENT_DIR/audio_samples/ || echo "無法列出文件"
    
    if [ $AUDIO_COUNT -gt 0 ]; then
        echo "🎉 ✅ 音頻樣本保存修復驗證成功!"
    else
        echo "❌ 音頻樣本保存仍然失敗"
    fi
else
    echo "❌ 音頻樣本目錄未創建"
fi

# 檢查模型檢查點
if [ -f "$EXPERIMENT_DIR/best_model.pth" ]; then
    echo "✅ 模型檢查點已保存"
else
    echo "❌ 模型檢查點未保存"
fi

echo ""
echo "📝 實驗總結:"
echo "🔧 修復內容: WavTokenizer decode 方法輸出從 2D 調整為 3D"
echo "🎯 測試目標: 驗證音頻樣本保存功能正常"
echo "⏰ 實驗完成時間: $(date)"
echo "📄 實驗記錄已保存到日誌文件"
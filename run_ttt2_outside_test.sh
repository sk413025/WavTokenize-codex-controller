#!/bin/bash
"""
TTT2 Outside音檔測試運行腳本
"""

echo "🎵 TTT2 Outside音檔測試"
echo "=========================="

# 檢查conda環境
if [[ "$CONDA_DEFAULT_ENV" != "test" ]]; then
    echo "⚠️  請先啟動conda環境: conda activate test"
    exit 1
fi

# 設置預設參數
CHECKPOINT_PATH="lightning_logs/version_0/checkpoints/epoch=299-step=300.ckpt"
OUTSIDE_DIR="outside_audio"
OUTPUT_DIR="ttt2_outside_test_results"
MAX_FILES=10
AUDIO_LENGTH=32000

# 檢查checkpoint是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT_PATH"
    echo "請檢查路徑或指定正確的checkpoint檔案"
    
    # 尋找可能的checkpoint檔案
    echo "尋找可能的checkpoint檔案..."
    find lightning_logs -name "*.ckpt" -type f 2>/dev/null | head -5
    exit 1
fi

# 檢查outside目錄是否存在
if [ ! -d "$OUTSIDE_DIR" ]; then
    echo "❌ Outside音檔目錄不存在: $OUTSIDE_DIR"
    echo "請創建目錄並放入測試音檔："
    echo "  mkdir -p $OUTSIDE_DIR"
    echo "  # 複製音檔到 $OUTSIDE_DIR/"
    exit 1
fi

# 檢查outside目錄中是否有音檔
AUDIO_COUNT=$(find "$OUTSIDE_DIR" -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.m4a" | wc -l)
if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "❌ 在 $OUTSIDE_DIR 中沒有找到音檔"
    echo "支援的格式: .wav, .mp3, .flac, .m4a"
    exit 1
fi

echo "✅ 找到 $AUDIO_COUNT 個音檔"
echo "✅ Checkpoint: $CHECKPOINT_PATH"

# 運行測試
echo ""
echo "開始TTT2 Outside音檔測試..."
python test_ttt2_outside.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --outside_dir "$OUTSIDE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_files "$MAX_FILES" \
    --audio_length "$AUDIO_LENGTH"

# 檢查結果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 測試完成！"
    echo "📁 結果目錄: $(find $OUTPUT_DIR -name "test_*" -type d | tail -1)"
    echo "📋 查看報告: $(find $OUTPUT_DIR -name "TEST_REPORT.md" | tail -1)"
else
    echo ""
    echo "❌ 測試失敗，請檢查錯誤信息"
fi

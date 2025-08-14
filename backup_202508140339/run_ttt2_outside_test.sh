#!/bin/bash
""# 設置預設參數
CHECKPOINT_PATH=""  # 將自動偵測
OUTSIDE_DIR="./1n"
OUTPUT_DIR="ttt2_outside_test_results"
MAX_FILES=10
AUDIO_LENGTH=32000Outside音檔測試運行腳本
"""

echo "🎵 TTT2 Outside音檔測試"
echo "=========================="

# 檢查conda環境
if [[ "$CONDA_DEFAULT_ENV" != "test" ]]; then
    echo "⚠️  請先啟動conda環境: conda activate test"
    exit 1
fi

# 設置預設參數
CHECKPOINT_PATH=""
OUTSIDE_DIR="outside_audio"
OUTPUT_DIR="ttt2_outside_test_results"
MAX_FILES=10
AUDIO_LENGTH=32000

# 自動尋找最佳checkpoint
echo "🔍 尋找可用的checkpoint..."

# 優先順序：best_model.pth > Lightning checkpoints
if [ -f "results/tsne_outputs/b-output4/best_model.pth" ]; then
    CHECKPOINT_PATH="results/tsne_outputs/b-output4/best_model.pth"
    echo "✅ 找到TTT2 best_model (b-output4): $CHECKPOINT_PATH"
elif [ -f "results/tsne_outputs/output4/best_model.pth" ]; then
    CHECKPOINT_PATH="results/tsne_outputs/output4/best_model.pth"
    echo "✅ 找到TTT2 best_model (output4): $CHECKPOINT_PATH"
elif [ -f "results/tsne_outputs/output3/best_model.pth" ]; then
    CHECKPOINT_PATH="results/tsne_outputs/output3/best_model.pth"
    echo "✅ 找到TTT2 best_model (output3): $CHECKPOINT_PATH"
else
    # 回退到Lightning checkpoint
    LIGHTNING_CKPT=$(find lightning_logs -name "*.ckpt" -type f | head -1)
    if [ ! -z "$LIGHTNING_CKPT" ]; then
        CHECKPOINT_PATH="$LIGHTNING_CKPT"
        echo "✅ 找到Lightning checkpoint: $CHECKPOINT_PATH"
    fi
fi

# 檢查checkpoint是否存在
if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 沒有找到可用的checkpoint"
    echo "請檢查以下路徑："
    echo "  - results/tsne_outputs/*/best_model.pth"
    echo "  - lightning_logs/*/checkpoints/*.ckpt"
    
    # 顯示可能的檔案
    echo ""
    echo "尋找可能的checkpoint檔案..."
    find results/tsne_outputs -name "best_model.pth" -type f 2>/dev/null
    find lightning_logs -name "*.ckpt" -type f 2>/dev/null | head -3
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

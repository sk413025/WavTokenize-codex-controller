#!/bin/bash

# ============================================================================
# 數據路徑驗證腳本
# 在開始訓練前運行此腳本，確保所有數據路徑都正確
# ============================================================================

echo "=========================================="
echo "驗證 EXP2 數據路徑"
echo "=========================================="
echo ""

# 切換到項目根目錄
cd "$(dirname "$0")"/../..

SUCCESS=true

# ============================================================================
# 檢查輸入資料夾
# ============================================================================
echo "1. 檢查輸入資料夾 (Noisy Audio):"
echo "-----------------------------------"

INPUT_DIRS=(
    "data/raw/box"
    "data/raw/papercup"
    "data/raw/plastic"
    "data/clean/box2"
)

for dir in "${INPUT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        count=$(ls "$dir"/*.wav 2>/dev/null | wc -l)
        echo "  ✓ $dir: 存在 ($count 個音檔)"

        # 顯示範例檔名
        example=$(ls "$dir"/*.wav 2>/dev/null | head -1)
        if [ -n "$example" ]; then
            echo "    範例: $(basename $example)"
        fi
    else
        echo "  ✗ $dir: 不存在"
        SUCCESS=false
    fi
done

echo ""

# ============================================================================
# 檢查目標資料夾
# ============================================================================
echo "2. 檢查目標資料夾 (Clean Audio):"
echo "-----------------------------------"

TARGET_DIR="data/clean/box2"

if [ -d "$TARGET_DIR" ]; then
    count=$(ls "$TARGET_DIR"/*.wav 2>/dev/null | wc -l)
    echo "  ✓ $TARGET_DIR: 存在 ($count 個音檔)"

    # 顯示範例檔名
    example=$(ls "$TARGET_DIR"/*.wav 2>/dev/null | head -1)
    if [ -n "$example" ]; then
        echo "    範例: $(basename $example)"
    fi
else
    echo "  ✗ $TARGET_DIR: 不存在"
    SUCCESS=false
fi

echo ""

# ============================================================================
# 檢查檔名格式
# ============================================================================
echo "3. 檢查檔名格式:"
echo "-----------------------------------"

# 檢查 noisy 檔名格式
noisy_example=$(ls data/raw/box/*.wav 2>/dev/null | head -1)
if [ -n "$noisy_example" ]; then
    filename=$(basename "$noisy_example")
    echo "  Noisy 檔名範例: $filename"

    # 解析檔名
    IFS='_' read -ra PARTS <<< "$filename"
    if [ ${#PARTS[@]} -ge 5 ]; then
        echo "    格式: ${PARTS[0]}_${PARTS[1]}_${PARTS[2]}_${PARTS[3]}_${PARTS[4]}"
        echo "    語者: ${PARTS[1]}"
        echo "    材質: ${PARTS[2]}"
        echo "    句子 ID: ${PARTS[4]%.wav}"
        echo "  ✓ 檔名格式正確"
    else
        echo "  ⚠️  檔名格式可能不符合預期"
    fi
fi

echo ""

# 檢查 clean 檔名格式
clean_example=$(ls data/clean/box2/*.wav 2>/dev/null | head -1)
if [ -n "$clean_example" ]; then
    filename=$(basename "$clean_example")
    echo "  Clean 檔名範例: $filename"

    # 解析檔名
    IFS='_' read -ra PARTS <<< "$filename"
    if [ ${#PARTS[@]} -ge 4 ]; then
        echo "    格式: ${PARTS[0]}_${PARTS[1]}_${PARTS[2]}_${PARTS[3]}"
        echo "    語者: ${PARTS[1]}"
        echo "    句子 ID: ${PARTS[3]%.wav}"
        echo "  ✓ 檔名格式正確"
    else
        echo "  ⚠️  檔名格式可能不符合預期"
    fi
fi

echo ""

# ============================================================================
# 統計語者分布
# ============================================================================
echo "4. 統計語者分布:"
echo "-----------------------------------"

# 統計 noisy 語者
echo "  Noisy 語者:"
for dir in "${INPUT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        speakers=$(ls "$dir"/*.wav 2>/dev/null | xargs -n1 basename | cut -d'_' -f2 | sort -u)
        echo "    $(basename $dir): $(echo "$speakers" | tr '\n' ' ')"
    fi
done

echo ""

# 統計 clean 語者
echo "  Clean 語者:"
speakers=$(ls "$TARGET_DIR"/*.wav 2>/dev/null | xargs -n1 basename | cut -d'_' -f2 | sort -u)
echo "    $(echo "$speakers" | tr '\n' ' ')"

echo ""

# ============================================================================
# 檢查 WavTokenizer 路徑
# ============================================================================
echo "5. 檢查 WavTokenizer 路徑:"
echo "-----------------------------------"

WAVTOKENIZER_CONFIG="/home/sbplab/ruizi/WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOKENIZER_CHECKPOINT="/home/sbplab/ruizi/WavTokenizer/results/smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_epoch_1200.pth"

if [ -f "$WAVTOKENIZER_CONFIG" ]; then
    echo "  ✓ Config: 存在"
else
    echo "  ✗ Config: 不存在"
    echo "    路徑: $WAVTOKENIZER_CONFIG"
    SUCCESS=false
fi

if [ -f "$WAVTOKENIZER_CHECKPOINT" ]; then
    size=$(du -h "$WAVTOKENIZER_CHECKPOINT" | cut -f1)
    echo "  ✓ Checkpoint: 存在 ($size)"
else
    echo "  ✗ Checkpoint: 不存在"
    echo "    路徑: $WAVTOKENIZER_CHECKPOINT"
    SUCCESS=false
fi

echo ""

# ============================================================================
# 檢查 Speaker Encoder
# ============================================================================
echo "6. 檢查 Speaker Encoder 依賴:"
echo "-----------------------------------"

# 檢查是否安裝 speechbrain
if python -c "import speechbrain" 2>/dev/null; then
    echo "  ✓ speechbrain: 已安裝"
else
    echo "  ⚠️  speechbrain: 未安裝（會在首次使用時自動下載 ECAPA-TDNN）"
    echo "    安裝指令: pip install speechbrain"
fi

# 檢查 speaker_encoder.py
if [ -f "done/exp/speaker_encoder.py" ]; then
    echo "  ✓ speaker_encoder.py: 存在"
else
    echo "  ✗ speaker_encoder.py: 不存在"
    SUCCESS=false
fi

echo ""

# ============================================================================
# 總結
# ============================================================================
echo "=========================================="
echo "驗證結果總結"
echo "=========================================="

if [ "$SUCCESS" = true ]; then
    echo "✅ 所有必要的數據路徑都存在！"
    echo ""
    echo "下一步："
    echo "  1. 運行測試腳本: python done/exp2/test_loss.py"
    echo "  2. 開始訓練: bash done/exp2/run_experiments.sh"
    echo "  或單獨訓練: python done/exp2/train_with_speaker.py ..."
else
    echo "❌ 部分路徑不存在，請修正後再繼續"
    echo ""
    echo "常見問題："
    echo "  1. 確認數據是否已準備好"
    echo "  2. 確認 WavTokenizer 是否已訓練"
    echo "  3. 確認所有路徑都正確"
fi

echo "=========================================="

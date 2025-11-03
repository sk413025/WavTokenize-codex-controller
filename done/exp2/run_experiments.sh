#!/bin/bash

# ============================================================================
# EXP2: 批次實驗腳本 - 對比不同 λ 值的 Speaker Loss
# ============================================================================

# 數據路徑（使用相對路徑，從 done/exp2/ 執行）
INPUT_DIRS=(
    "../../data/raw/box"
    "../../data/raw/papercup"
    "../../data/raw/plastic"
    "../../data/clean/box2"
)
TARGET_DIR="../../data/clean/box2"

# 實驗參數
NUM_EPOCHS=600
BATCH_SIZE=8
MAX_SENTENCES=288
SPEAKER_MODEL="ecapa"

# 將 INPUT_DIRS 陣列轉換為字串參數
INPUT_DIRS_STR="${INPUT_DIRS[0]} ${INPUT_DIRS[1]}"

echo "=========================================="
echo "EXP2: Baseline + Speaker Loss 實驗"
echo "=========================================="
echo "數據集: nor_boy1_10_girl1_11"
echo "輸入目錄: ${INPUT_DIRS[*]}"
echo "目標目錄: $TARGET_DIR"
echo "訓練輪數: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# ============================================================================
# 實驗 1: λ = 0.1 (輕量約束)
# ============================================================================
echo "=========================================="
echo "實驗 1: λ = 0.1 (輕量約束)"
echo "=========================================="

python done/exp2/train_with_speaker.py \
    --input_dirs $INPUT_DIRS_STR \
    --target_dir "$TARGET_DIR" \
    --output_dir "./results/exp2/lambda0.1" \
    --lambda_speaker 0.1 \
    --speaker_model_type "$SPEAKER_MODEL" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_sentences_per_speaker "$MAX_SENTENCES"

echo ""
echo "✓ 實驗 1 完成"
echo ""

# ============================================================================
# 實驗 2: λ = 0.5 (中等約束 - 推薦)
# ============================================================================
echo "=========================================="
echo "實驗 2: λ = 0.5 (中等約束 - 推薦)"
echo "=========================================="

python done/exp2/train_with_speaker.py \
    --input_dirs $INPUT_DIRS_STR \
    --target_dir "$TARGET_DIR" \
    --output_dir "./results/exp2/lambda0.5" \
    --lambda_speaker 0.5 \
    --speaker_model_type "$SPEAKER_MODEL" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_sentences_per_speaker "$MAX_SENTENCES"

echo ""
echo "✓ 實驗 2 完成"
echo ""

# ============================================================================
# 實驗 3: λ = 1.0 (強約束)
# ============================================================================
echo "=========================================="
echo "實驗 3: λ = 1.0 (強約束)"
echo "=========================================="

python done/exp2/train_with_speaker.py \
    --input_dirs $INPUT_DIRS_STR \
    --target_dir "$TARGET_DIR" \
    --output_dir "./results/exp2/lambda1.0" \
    --lambda_speaker 1.0 \
    --speaker_model_type "$SPEAKER_MODEL" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_sentences_per_speaker "$MAX_SENTENCES"

echo ""
echo "✓ 實驗 3 完成"
echo ""

# ============================================================================
# 總結
# ============================================================================
echo "=========================================="
echo "所有實驗完成！"
echo "=========================================="
echo ""
echo "結果目錄:"
echo "  - ./results/exp2/lambda0.1/"
echo "  - ./results/exp2/lambda0.5/"
echo "  - ./results/exp2/lambda1.0/"
echo ""
echo "下一步："
echo "  1. 檢查各實驗的 training.log"
echo "  2. 對比 loss_curves_epoch_XXX.png"
echo "  3. 聆聽 audio_samples/ 中的降噪效果"
echo "  4. 將結果填入 done/exp2/README.md 的實驗記錄表格"
echo ""

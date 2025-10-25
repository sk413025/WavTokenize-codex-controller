#!/bin/bash

# 快速測試修復 - 只執行 2 個 epoch
# 驗證所有修復是否正常工作

set -e

cd "$(dirname "$0")"

# 測試輸出目錄
TEST_OUTPUT="../results/test_fixes_$(date +%Y%m%d_%H%M%S)"

echo "======================================"
echo "快速測試修復後的訓練腳本"
echo "只執行 2 個 epoch 以驗證所有修復"
echo "======================================"
echo "測試項目："
echo "1. ✓ 訓練是否正常執行"
echo "2. ✓ 維度處理是否正確"
echo "3. ✓ checkpoint 是否正常儲存"
echo "4. ✓ 新參數配置是否生效"
echo "======================================"
echo ""

# 自動選擇空閒的GPU
echo "🔍 檢測可用的GPU..."
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    awk '$2 > 8000 && $1 != 2 {print $1}' | head -1)

if [ -z "$AVAILABLE_GPUS" ]; then
    echo "⚠️  沒有找到足夠空閒的GPU，使用預設GPU"
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS
    echo "✅ 使用 GPU: $AVAILABLE_GPUS"
fi

echo ""
echo "開始測試..."
echo ""

python -u train_token_denoising_hybrid.py \
    --input_dirs ../data/raw/box \
    --target_dir ../data/clean/box2 \
    --output_dir ${TEST_OUTPUT} \
    --batch_size 4 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --d_model 512 \
    --nhead 8 \
    --num_layers 4 \
    --dropout 0.2 \
    --ce_weight 1.0 \
    --content_weight 0.5 \
    --embed_weight 0.3 \
    --warmup_epochs 1 \
    --content_ratio 0.5 \
    --min_content_samples 3 \
    --max_sentences_per_speaker 288 \
    --wavtokenizer_config ../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --wavtokenizer_checkpoint ../models/wavtokenizer_large_speech_320_24k.ckpt

echo ""
echo "======================================"
echo "✅ 測試完成！"
echo "======================================"
echo "輸出目錄: ${TEST_OUTPUT}"
echo ""
echo "檢查清單："
echo "  [ ] 訓練正常完成（無錯誤）"
echo "  [ ] checkpoint 已儲存"
echo "  [ ] 配置檔案已生成"
echo "  [ ] 新參數配置已生效（dropout=0.2, num_layers=4）"
echo ""
echo "如果所有檢查都通過，可以執行完整訓練："
echo "  bash run_fixed_training.sh"
echo "===================================="

#!/bin/bash

# WavTokenizer-Transformer 降噪訓練 - Token Loss 修復後恢復訓練
echo "🔧 修復後重新啟動 Token Loss 訓練"
echo "====================================================="
echo "   - 修復: tensor view -> reshape 避免CUDA錯誤"
echo "   - 恢復點: model_epoch_50.pth (第50個epoch)"
echo "   - 剩餘訓練: 550 epochs (50→600)"
echo "   - 新實驗編號: tokenloss_fixed_$(date +%Y%m%d%H%M)"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TTT_BATCH_SIZE=8
export TTT_NUM_WORKERS=4
export CONTENT_BATCHING=true
export CUDA_LAUNCH_BLOCKING=1         # 調試CUDA錯誤

set -e

# 獲取實驗編號和時間戳
EXP_ID="tokenloss_fixed_$(date +%Y%m%d%H%M)"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
LOG_FILE="logs/${EXP_ID}.log"
OUTPUT_DIR="results/wavtokenizer_tokenloss_${EXP_ID}"

# 源檢查點路徑
SOURCE_CHECKPOINT="./results/wavtokenizer_tokenloss_wavtok_tokenloss_202509180534/model_epoch_50.pth"

echo ""
echo "🚀 WavTokenizer-Transformer Token Loss 修復恢復訓練"
echo "====================================================="
echo "實驗編號: $EXP_ID"
echo "開始時間: $TIMESTAMP"
echo "輸出目錄: $OUTPUT_DIR"
echo "日誌文件: $LOG_FILE"
echo "源檢查點: $SOURCE_CHECKPOINT"
echo ""

# 檢查源檢查點是否存在
if [ ! -f "$SOURCE_CHECKPOINT" ]; then
    echo "❌ 錯誤: 源檢查點不存在: $SOURCE_CHECKPOINT"
    exit 1
fi

echo "✅ 源檢查點確認存在"

# 創建輸出目錄
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

echo ""
echo "🔧 環境變數設定:"
echo "   ONLY_USE_BOX_MATERIAL: $ONLY_USE_BOX_MATERIAL"
echo "   TTT_BATCH_SIZE: $TTT_BATCH_SIZE"
echo "   TTT_NUM_WORKERS: $TTT_NUM_WORKERS"
echo "   CONTENT_BATCHING: $CONTENT_BATCHING"
echo "   CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING (CUDA調試)"
echo ""

echo "🎯 恢復訓練配置："
echo "1. ✅ 從第50個epoch恢復"
echo "2. ✅ 修復後的Token Loss系統 (reshape取代view)"
echo "3. ✅ CUDA調試模式啟用"
echo "4. ✅ 繼續訓練至600 epochs"
echo ""

echo "⚠️  注意: 確保其他GPU實驗不受影響"
echo ""

# 啟動修復後的訓練
echo "🚀 啟動修復後的Token Loss訓練..."
echo "開始時間: $(date)"

python wavtokenizer_transformer_denoising.py \
    --config config/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --model_path /home/sbplab/ruizi/WavTokenizer/results/exp_smalldata_75/model_checkpoint.pth \
    --output_dir $OUTPUT_DIR \
    --use_token_loss \
    --l2_weight 0.3 \
    --consistency_weight 0.4 \
    --manifold_weight 0.1 \
    --normalization_weight 0.1 \
    --coherence_weight 0.1 \
    --batch_size 8 \
    --num_epochs 600 \
    --learning_rate 1e-4 \
    --max_length 200 \
    --save_every 50 \
    --val_speakers girl9 boy7 \
    --max_sentences_per_speaker 100 \
    --resume_from_checkpoint $SOURCE_CHECKPOINT \
    2>&1 | tee -a $LOG_FILE

# 檢查訓練結果
TRAIN_EXIT_CODE=$?
echo ""
echo "====================================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 修復後的Token Loss訓練成功完成！"
    echo "🎯 關鍵修復點:"
    echo "   - tensor.view() → tensor.reshape() 修復CUDA錯誤"
    echo "   - 從第50個epoch成功恢復"
    echo "   - 啟用CUDA調試模式"
else
    echo "❌ 修復後的訓練仍有問題，退出碼: $TRAIN_EXIT_CODE"
    echo "📋 檢查日誌: $LOG_FILE"
fi

echo ""
echo "📊 訓練統計:"
echo "結束時間: $(date)"
echo "日誌文件: $LOG_FILE"
echo "輸出目錄: $OUTPUT_DIR"
echo "====================================================="

exit $TRAIN_EXIT_CODE

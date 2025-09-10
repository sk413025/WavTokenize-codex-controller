#!/bin/bash

# WavTokenizer-Transformer 降噪訓練 - Token Loss 系統模式
echo "   - 總參數: 89.3M (80.6M凍結 + 8.7M可訓練)"
echo "   - Token Loss: 主導損失函數 (90%)"
echo "   - CrossEntropy: 輔助損失 (10%)"
echo "====================================================="

# 設置環境變數 (與 ttt2.py 一致)
export ONLY_USE_BOX_MATERIAL=true     # 僅處理 box 材質
export PYTHONUNBUFFERED=1             # 即時輸出日誌
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # CUDA記憶體配置
export TTT_BATCH_SIZE=8               # 批次大小 (與 run_fixed_ttt2_branch.sh 一致)
export TTT_NUM_WORKERS=4              # 資料載入工作線程數
export CONTENT_BATCHING=true          # 啟用內容感知批次採樣

echo ""
echo "🔧 環境變數設定:"
echo "   ONLY_USE_BOX_MATERIAL: $ONLY_USE_BOX_MATERIAL"
echo "   TTT_BATCH_SIZE: $TTT_BATCH_SIZE"
echo "   TTT_NUM_WORKERS: $TTT_NUM_WORKERS"
echo "   CONTENT_BATCHING: $CONTENT_BATCHING"ormer 降噪訓練 - Token Loss 系統模式
# 實驗編號：EXP-WAVTOK-TL-$(date +%Y%m%d%H%M)

set -e  # 腳本遇到錯誤時停止運行

# 獲取實驗編號和時間戳
EXP_ID="wavtok_tokenloss_$(date +%Y%m%d%H%M)"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
LOG_FILE="logs/${EXP_ID}.log"
OUTPUT_DIR="results/wavtokenizer_tokenloss_${EXP_ID}"

echo "====================================================="
echo "🚀 WavTokenizer-Transformer - Advanced Token Loss"
echo "====================================================="
echo "實驗編號: $EXP_ID"
echo "開始時間: $TIMESTAMP"
echo "輸出目錄: $OUTPUT_DIR"
echo "日誌文件: $LOG_FILE"
echo ""
echo "🎯 系統架構："
echo "1. ✅ Audio → WavTokenizer Encoder (凍結)"
echo "2. ✅ Token → Transformer Denoiser (可訓練)"
echo "3. ✅ Token → WavTokenizer Decoder (凍結)"
echo "4. ✅ 高級 Token Loss 系統 (基於 ttt2.py)"
echo ""
echo "� Token Loss 系統組件："
echo "   - L2 距離損失 (30%): Token 嵌入空間對齊"
echo "   - 內容一致性損失 (40%): 相同內容 ID 的 token 一致性"
echo "   - Manifold 正則化 (10%): Token 流形約束"
echo "   - 正則化損失 (10%): Token 分佈正則化" 
echo "   - 連貫性損失 (10%): Token 序列連貫性"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true     # 僅處理 box 材質
export PYTHONUNBUFFERED=1             # 即時輸出日誌
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # CUDA記憶體配置
export TTT_BATCH_SIZE=8               # 批次大小
export TTT_NUM_WORKERS=4              # 資料載入工作線程數
export CONTENT_BATCHING=true          # 啟用內容感知批次採樣

echo ""
echo "🔧 環境變數設定:"
echo "   ONLY_USE_BOX_MATERIAL: $ONLY_USE_BOX_MATERIAL"
echo "   TTT_BATCH_SIZE: $TTT_BATCH_SIZE" 
echo "   TTT_NUM_WORKERS: $TTT_NUM_WORKERS"
echo "   CONTENT_BATCHING: $CONTENT_BATCHING"

# 創建必要目錄
mkdir -p logs
mkdir -p results
mkdir -p $OUTPUT_DIR

echo ""
echo "📁 目錄設定完成"

# 檢查必要文件
CONFIG_FILE="config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
MODEL_FILE="models/wavtokenizer_large_speech_320_24k.ckpt"

echo ""
echo "🔍 檢查必要文件..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 錯誤: 找不到配置文件 $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ 錯誤: 找不到模型文件 $MODEL_FILE"
    exit 1
fi

if [ ! -f "wavtokenizer_transformer_denoising.py" ]; then
    echo "❌ 錯誤: 找不到訓練腳本 wavtokenizer_transformer_denoising.py"
    exit 1
fi

if [ ! -f "token_loss_system.py" ]; then
    echo "❌ 錯誤: 找不到 Token Loss 系統 token_loss_system.py"
    exit 1
fi

echo "✅ 所有必要文件檢查完成"

# 激活 conda 環境（如果存在）
echo ""
echo "🐍 激活 conda 環境..."
if command -v conda &> /dev/null; then
    source /home/sbplab/miniconda3/etc/profile.d/conda.sh
    if conda env list | grep -q "test"; then
        conda activate test
        echo "✅ 已激活 conda test 環境"
    else
        echo "⚠️  使用當前環境（test 環境未找到）"
    fi
else
    echo "⚠️  Conda 不可用，使用當前 Python 環境"
fi

# 檢查 GPU
echo ""
echo "🚀 檢查 GPU 可用性..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA 可用: {torch.cuda.device_count()} 個 GPU')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA 不可用，將使用 CPU')
"

# 清理 CUDA 緩存
echo ""
echo "🧹 清理 CUDA 緩存..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "無法清空 CUDA 緩存"

echo ""
echo "🚀 開始訓練 - Token Loss 系統模式"
echo "⏰ 開始時間: $(date)"
echo ""

# 執行訓練 - Token Loss 模式（使用 --use_token_loss）
# 參數設定與 ttt2.py 一致: batch_size=8, 語者分類, 每位語者100句話, 僅box材質
python wavtokenizer_transformer_denoising.py \
    --output_dir "$OUTPUT_DIR" \
    --use_token_loss \
    --l2_weight 0.3 \
    --consistency_weight 0.4 \
    --manifold_weight 0.1 \
    --normalization_weight 0.1 \
    --coherence_weight 0.1 \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --max_length 200 \
    --data_dir data \
    --save_interval 10 \
    --log_interval 5 \
    --val_speakers girl9 boy7 \
    --max_sentences_per_speaker 100 \
    2>&1 | tee -a $LOG_FILE

# 檢查訓練結果
TRAIN_EXIT_CODE=$?
echo ""
echo "====================================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "🎉 WavTokenizer-Transformer Token Loss 訓練完成！"
    echo "📊 訓練成功結束時間: $(date)"
    echo "📁 結果保存位置: $OUTPUT_DIR"
    echo "📊 訓練日誌: $LOG_FILE"
    
    # 檢查輸出文件
    echo ""
    echo "📂 檢查訓練輸出："
    if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
        echo "✅ 最佳模型: $OUTPUT_DIR/best_model.pth"
    fi
    if [ -f "$OUTPUT_DIR/final_model.pth" ]; then
        echo "✅ 最終模型: $OUTPUT_DIR/final_model.pth"
    fi
    if [ -f "$OUTPUT_DIR/training_history.png" ]; then
        echo "✅ 訓練曲線: $OUTPUT_DIR/training_history.png"
    fi
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        echo "✅ 詳細日誌: $OUTPUT_DIR/training.log"
    fi
    
    echo ""
    echo "🎯 實驗總結："
    echo "   架構: Audio → WavTokenizer → Transformer → Audio"
    echo "   損失: Advanced Token Loss System (基於 ttt2.py)"
    echo "   參數: 89.3M總參數 (80.6M凍結 + 8.7M可訓練)"
    echo "   組件: L2(30%) + 一致性(40%) + Manifold(10%) + 正規化(10%) + 連貫性(10%)"
    echo "   特點: 端到端音頻降噪 + 高級損失約束"
    
else
    echo "❌ 訓練失敗！退出碼: $TRAIN_EXIT_CODE"
    echo "📊 錯誤時間: $(date)"
    echo "📋 請檢查日誌文件: $LOG_FILE"
fi

echo "====================================================="
echo "實驗編號: $EXP_ID"
echo "結束時間: $(date)"
echo "====================================================="

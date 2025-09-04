#!/bin/bash

# 獲取當前日期時間作為實驗編號
EXP_ID=$(date +%Y%m%d%H%M)
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "====================================================="
echo "開始執行內容一致性損失對比實驗 - $EXP_ID"
echo "====================================================="
echo "實驗一：階層式內容一致性損失（連續+離散特徵）"
echo "實驗二：純離散內容一致性損失"
echo ""

# 激活 conda 環境
echo "激活 conda test 環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export TTT_BATCH_SIZE=8
export TTT_NUM_WORKERS=4

# 檢查環境
echo "當前 conda 環境: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import einops; print('✅ einops 可用')"

# 創建輸出目錄
mkdir -p results/tsne_outputs/exp-hierarchical-${EXP_ID}
mkdir -p results/tsne_outputs/exp-discrete-${EXP_ID}
mkdir -p logs

echo ""
echo "🚀 啟動實驗一：階層式內容一致性損失"
echo "輸出目錄: results/tsne_outputs/exp-hierarchical-${EXP_ID}"

# 啟動階層式實驗（後台）
python ttt2.py \
    --experiment_hierarchical_content \
    --hierarchy_alpha 0.7 \
    --content_alpha 0.01 \
    --save_dir results/tsne_outputs/exp-hierarchical-${EXP_ID} \
    2>&1 | tee logs/hierarchical_${EXP_ID}.log &

HIERARCHICAL_PID=$!
echo "階層式實驗 PID: $HIERARCHICAL_PID"

# 等待一會兒讓第一個實驗啟動
sleep 30

echo ""
echo "🚀 啟動實驗二：純離散內容一致性損失"
echo "輸出目錄: results/tsne_outputs/exp-discrete-${EXP_ID}"

# 啟動純離散實驗（後台）
python ttt2.py \
    --experiment_discrete_content \
    --content_alpha 0.01 \
    --save_dir results/tsne_outputs/exp-discrete-${EXP_ID} \
    2>&1 | tee logs/discrete_${EXP_ID}.log &

DISCRETE_PID=$!
echo "純離散實驗 PID: $DISCRETE_PID"

echo ""
echo "====================================================="
echo "兩個實驗已啟動："
echo "1. 階層式實驗 PID: $HIERARCHICAL_PID"
echo "2. 純離散實驗 PID: $DISCRETE_PID"
echo ""
echo "監控指令："
echo "- 查看階層式實驗: tail -f logs/hierarchical_${EXP_ID}.log"
echo "- 查看純離散實驗: tail -f logs/discrete_${EXP_ID}.log"
echo "- 檢查進程狀態: ps aux | grep python"
echo ""
echo "實驗預計運行時間: 6-12 小時"
echo "======================================================"

# 等待用戶確認
read -p "按 Enter 繼續監控實驗，或 Ctrl+C 退出..."

# 監控實驗狀態
while true; do
    echo ""
    echo "===== 實驗狀態監控 $(date) ====="
    
    # 檢查進程是否仍在運行
    if ps -p $HIERARCHICAL_PID > /dev/null; then
        echo "✅ 階層式實驗仍在運行 (PID: $HIERARCHICAL_PID)"
    else
        echo "❌ 階層式實驗已結束 (PID: $HIERARCHICAL_PID)"
    fi
    
    if ps -p $DISCRETE_PID > /dev/null; then
        echo "✅ 純離散實驗仍在運行 (PID: $DISCRETE_PID)"
    else
        echo "❌ 純離散實驗已結束 (PID: $DISCRETE_PID)"
    fi
    
    # 檢查 GPU 使用情況
    echo ""
    echo "GPU 使用情況："
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "無法獲取 GPU 信息"
    
    sleep 300  # 每5分鐘檢查一次
done

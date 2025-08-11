#!/bin/bash

# 確保腳本在錯誤時會停止運行
set -e

# 顯示腳本運行訊息
echo "====================================================="
echo "開始執行 TTT 模型 (純 L2 損失模式 + 僅使用 box 材質)"
echo "====================================================="
echo "參數設定:"
echo "1. --tsne_flow_with_L2: 處理流程與 tsne.py 完全一致，只使用 L2 損失函數"
echo "2. ONLY_USE_BOX_MATERIAL=true: 僅使用 box 材質數據"
echo "====================================================="

# 設置環境變數，指示只處理 box 材質
export ONLY_USE_BOX_MATERIAL=true

# 運行模型
# 使用批次大小為4以節省記憶體，並加入額外記憶體監控
echo "注意: 已將批次大小設為4以避免記憶體不足問題"
echo "====================================================="

# 設置較低的批次大小和限制CUDA記憶體增長
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 運行前清理CUDA緩存
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 執行 ttt.py 使用與 tsne.py 相同的邏輯 (只計算 L2 損失)
python ttt.py --tsne_flow_with_L2

# 顯示完成訊息
echo ""
echo "====================================================="
echo "程序執行完成"
echo "====================================================="

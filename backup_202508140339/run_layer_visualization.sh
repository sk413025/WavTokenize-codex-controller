#!/bin/bash

# 設置腳本錯誤時停止運行
set -e

# 獲取當前日期時間作為實驗編號
EXP_ID=$(date +%Y%m%d%H%M)
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 顯示腳本運行訊息
echo "====================================================="
echo "執行分層特徵可視化分析 - $EXP_ID"
echo "====================================================="
echo "功能描述: 收集並可視化模型各層殘差塊的特徵空間演變過程"
echo "可視化項目:"
echo "1. 各殘差層的特徵分布 t-SNE 圖"
echo "2. 特徵變化過程動態可視化"
echo "3. 層間特徵距離變化分析"
echo "====================================================="

# 設置環境變數
export PYTHONUNBUFFERED=1  # 即時輸出日誌
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 限制CUDA記憶體分配塊大小
export TTT_BATCH_SIZE=4    # 較小批次大小以節省記憶體
export TTT_NUM_WORKERS=4   # 資料載入工作線程數
export TTT_EXPERIMENT_ID="${EXP_ID}_layer_viz"  # 設置實驗ID
export VISUALIZE_ALL_LAYERS=true  # 啟用所有層的可視化
# 注意：3D t-SNE 功能已移除

# 設置日誌和輸出目錄
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/layer_viz_${EXP_ID}.log"

echo "執行特徵可視化..."
python feature_layer_visualization.py 2>&1 | tee $LOG_FILE

# 更新報告
cat << EOF >> REPORT.md

## ${TIMESTAMP} 分層特徵空間可視化分析 (EXP${EXP_ID})
- **分析類型**: 層級特徵演變可視化
- **分析目的**: 觀察各殘差塊層的特徵空間變化，理解模型內部表示演變
- **可視化保存路徑**: \`results/layer_viz_${EXP_ID}/\`
- **分析結果摘要**:
  - 各層特徵空間的演變過程
  - 特徵空間從語義表示到聲學表示的過渡情況
  - 內容一致性損失和L2損失對特徵分布的影響
- **增強可視化**:
  - 特徵空間演變動畫展示
  - 各層特徵與目標特徵的空間分佈關係分析
EOF

echo "====================================================="
echo "特徵可視化分析完成，結果保存在 results/layer_viz_${EXP_ID}/"
echo "日誌文件: ${LOG_FILE}"
echo "====================================================="

#!/bin/bash

# 特徵漂移監控腳本
# 日期: 2025-07-01

# 檢查參數
if [ "$#" -lt 1 ]; then
  echo "用法: $0 <實驗目錄> [輸出目錄]"
  echo "例如: $0 results/mixed_loss_experiment/EXP20250701_123456"
  exit 1
fi

EXPERIMENT_DIR=$1
OUTPUT_DIR=${2:-"${EXPERIMENT_DIR}/analysis"}

# 創建輸出目錄
mkdir -p "$OUTPUT_DIR"

echo "分析特徵漂移: ${EXPERIMENT_DIR}"
echo "輸出目錄: ${OUTPUT_DIR}"

# 執行特徵漂移分析
python feature_drift_monitor.py analyze \
  --experiment_dir "$EXPERIMENT_DIR" \
  --output_dir "$OUTPUT_DIR"

echo "分析完成。結果保存在: ${OUTPUT_DIR}"
echo "報告路徑: ${OUTPUT_DIR}/feature_drift_report.md"

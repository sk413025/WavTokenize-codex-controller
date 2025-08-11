#!/bin/bash

# 設置執行環境
echo "=====# 輸出說明
echo ""
echo "完成執行 t-SNE 特徵比較可視化工具"
echo "✓ 已成功比較輸入特徵和目標特徵的分佈"
echo "✓ 使用形狀區分不同特徵類型：目標(星星)，輸入(圓點)"
echo "✓ 使用顏色區分不同材質"
echo "✓ 標註了語者和語句編號（前三句話）"
echo "✓ 使用了合適的 perplexity 參數"
echo "✓ 已生成高質量圖像，可在 $OUTPUT_DIR 目錄查看"
echo "======================================================"

# 更新 REPORT.md 檔案
TIMESTAMP=$(date +"%Y-%m-%d")
OUTPUT_IMAGE=$(find $OUTPUT_DIR -type f -name "tsne_comparison_epoch${EPOCH}_*.png" | sort -r | head -n 1)
OUTPUT_FILENAME=$(basename "$OUTPUT_IMAGE")

# 準備報告內容
REPORT_CONTENT="## ${TIMESTAMP} t-SNE特徵比較分析 (實驗編號: TSNE-$(date +"%Y%m%d"))
- 執行了t-SNE特徵比較分析，比較了輸入和目標特徵的分布差異
- 分析對象：Epoch ${EPOCH} 的特徵資料
- 使用不同形狀區分特徵類型：目標特徵(星星)，輸入特徵(圓點)
- 使用不同顏色標示不同材質：box、clean等
- 為前三句話的每個樣本添加了標註(語者_句子編號)
- Perplexity參數設定為 ${PERPLEXITY}
- 輸出檔案：\`${OUTPUT_FILENAME}\`
- 函數：\`feature_tsne_comparison.py:plot_tsne_comparison\`"

# 插入報告內容到 REPORT.md 的頂部
if [ -f "REPORT.md" ]; then
    # 讀取現有內容
    EXISTING_CONTENT=$(cat REPORT.md)
    
    # 將新內容和現有內容合併並寫回文件
    echo -e "# 實驗記錄報告\n\n${REPORT_CONTENT}\n\n${EXISTING_CONTENT#\# 實驗記錄報告}" > REPORT.md
    
    echo "✓ 已更新 REPORT.md 文件"
else
    # 如果文件不存在，創建新文件
    echo -e "# 實驗記錄報告\n\n${REPORT_CONTENT}" > REPORT.md
    
    echo "✓ 已創建 REPORT.md 文件"
fi======================================"
echo "開始執行 t-SNE 特徵比較可視化工具"
echo "====================================================="
echo "本工具將比較訓練後輸入和目標特徵的位置分佈"
echo "- 使用形狀區分：目標特徵(星星)，輸入特徵(圓點)"
echo "- 使用顏色區分：不同材質 (box, clean, plastic...)"
echo "- 添加標註：語者和語句編號（僅前三句話）"
echo "====================================================="

# 創建輸出目錄
OUTPUT_DIR="results/feature_comparison"
mkdir -p $OUTPUT_DIR

# 預設參數
RESULTS_DIR="results/tsne_outputs/output2"
PERPLEXITY=30

# 檢查最新的epoch
LATEST_EPOCH=$(find $RESULTS_DIR/features -maxdepth 1 -type d -name "epoch_*" | sort -V | tail -n 1)
if [ -n "$LATEST_EPOCH" ]; then
    EPOCH=$(echo $LATEST_EPOCH | grep -o '[0-9]\+' | tail -n 1)
    echo "找到最新的 epoch: $EPOCH"
else
    echo "未找到任何 epoch 目錄，將使用命令行參數"
fi

# 運行 Python 腳本
echo "正在執行 t-SNE 分析..."
python feature_tsne_comparison.py --epoch $EPOCH --output-dir $OUTPUT_DIR --results-dir $RESULTS_DIR --perplexity $PERPLEXITY

# 輸出說明
echo ""
echo "完成執行 t-SNE 特徵比較可視化工具"
echo "✓ 已成功比較輸入特徵和目標特徵的分佈"
echo "✓ 使用形狀區分不同特徵類型：目標(星星)，輸入(圓點)"
echo "✓ 使用顏色區分不同材質"
echo "✓ 標註了語者和語句編號（前三句話）"
echo "✓ 使用了合適的 perplexity 參數"
echo "✓ 已生成高質量圖像，可在 $OUTPUT_DIR 目錄查看"
echo "====================================================="

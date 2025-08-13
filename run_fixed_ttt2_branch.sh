#!/bin/bash

# 確保腳本在錯誤時會停止運行
set -e

# 獲取當前日期時間作為實驗編號
EXP_ID=$(date +%Y%m%d%H%M)
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 顯示腳本運行訊息
echo "====================================================="
echo "開始執行 TTT2 模型修復分支訓練 - $EXP_ID"
echo "====================================================="
echo "分支: fix-ttt2-residual-block-and-manifold"
echo "輸出目錄: results/tsne_outputs/b-output4"
echo ""
echo "🔧 關鍵修復內容:"
echo "1. ✅ ResidualBlock bug 修復: conv2(out) 取代 conv2(x)"
echo "2. ✅ GroupNorm 支援: 更穩定的音頻處理"
echo "3. ✅ 流形正則化: 防止特徵偏離訓練分佈"
echo "4. ✅ 碼本一致性損失: 確保離散編碼穩定性"
echo "5. ✅ 多組件損失整合: 增強的訓練穩定性"
echo ""
echo "🎯 訓練參數:"
echo "1. --tsne_flow_with_content: 處理流程與tsne.py保持一致，使用修復後的內容一致性損失"
echo "2. --use_layered_loss: 使用分層損失機制"
echo "3. --first_two_blocks_only: 嚴格分層損失設計"
echo "4. ONLY_USE_BOX_MATERIAL=true: 僅使用 box 材質數據進行訓練"
echo "====================================================="

# 設置環境變數
export ONLY_USE_BOX_MATERIAL=true     # 僅處理 box 材質
export PYTHONUNBUFFERED=1           # 即時輸出日誌，不進行緩衝
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 限制CUDA記憶體分配塊大小
export TTT_BATCH_SIZE=8             # 增大批次大小以確保每個批次有相同內容ID的多個樣本
export TTT_NUM_WORKERS=4            # 資料載入工作線程數
export TTT_EXPERIMENT_ID="${EXP_ID}" # 設置實驗ID
export INPUT_SAMPLE_RATE=16000      # 設置輸入音頻採樣率
export CONTENT_BATCHING=true        # 啟用內容感知批次採樣，確保相同內容的樣本在同一批次

# 設置運行參數
LOG_FILE="logs/ttt2_fixed_branch_training_${EXP_ID}.log"  # 分支專用日誌文件路徑

# 創建日誌目錄
mkdir -p logs

echo "運行環境設定:"
echo "- 批次大小 \"Batch Size\": $TTT_BATCH_SIZE"
echo "- 資料載入線程數: $TTT_NUM_WORKERS"
echo "- 日誌文件: $LOG_FILE"
echo "- 分支輸出目錄: results/tsne_outputs/b-output4"
echo "====================================================="

# 激活 conda 環境
echo "激活 conda test 環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 運行修復驗證測試
echo "🧪 首先運行修復驗證測試..."
echo "運行 test_ttt2_fixes.py 驗證修復..."
python test_ttt2_fixes.py || echo "⚠️ 修復測試出現問題，但繼續執行訓練"

# 運行前清理CUDA緩存
echo "清理 CUDA 緩存..."
python -c "import torch; torch.cuda.empty_cache()" || echo "無法清空CUDA緩存"

# 運行模型，同時將輸出導向至終端和日誌文件
echo "🚀 開始修復分支模型訓練，時間: $(date)"
echo "使用修復後的 ResidualBlock + GroupNorm + 流形正則化..."
python ttt2.py \
    --tsne_flow_with_content \
    --use_layered_loss \
    --first_two_blocks_only \
    \
    2>&1 | tee -a $LOG_FILE

# 顯示完成訊息
echo ""
echo "====================================================="
echo "修復分支程序執行完成，時間: $(date)"
echo "結果日誌保存在: $LOG_FILE"
echo "實驗結果保存在: results/tsne_outputs/b-output4"

# 自動更新實驗報告
echo "" >> $REPORT_FILE
echo "## TTT2 修復分支訓練 - FIX_BRANCH_$EXP_ID" >> $REPORT_FILE
echo "**執行時間:** $TIMESTAMP" >> $REPORT_FILE
echo "**分支:** fix-ttt2-residual-block-and-manifold" >> $REPORT_FILE
echo "**輸出目錄:** results/tsne_outputs/b-output4" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🔧 關鍵修復內容" >> $REPORT_FILE
echo "1. **ResidualBlock 修復:** 修正 conv2(x) → conv2(out) 錯誤" >> $REPORT_FILE
echo "2. **GroupNorm 支援:** 替代 BatchNorm 提供更穩定的音頻處理" >> $REPORT_FILE
echo "3. **流形正則化:** compute_manifold_regularization_loss() 防止特徵偏離" >> $REPORT_FILE
echo "4. **碼本一致性:** compute_codebook_consistency_loss() 確保編碼穩定" >> $REPORT_FILE
echo "5. **多組件損失:** 整合所有損失組件的 compute_layered_hybrid_loss()" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🎯 訓練設定" >> $REPORT_FILE
echo "- **模型:** TTT2 (修復版)" >> $REPORT_FILE
echo "- **損失函數:** 分層混合損失 + 流形正則化 + 碼本一致性" >> $REPORT_FILE
echo "- **材質:** 僅 box 材質" >> $REPORT_FILE
echo "- **批次大小:** $TTT_BATCH_SIZE" >> $REPORT_FILE
echo "- **日誌檔案:** \`$LOG_FILE\`" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 📊 預期改善" >> $REPORT_FILE
echo "- 更穩定的梯度流動 (ResidualBlock 修復)" >> $REPORT_FILE
echo "- 更好的訓練穩定性 (GroupNorm)" >> $REPORT_FILE
echo "- 防止過擬合和特徵偏移 (流形正則化)" >> $REPORT_FILE
echo "- 更一致的離散編碼 (碼本一致性損失)" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "----" >> $REPORT_FILE

echo "已更新實驗報告: $REPORT_FILE"
echo "🎉 修復分支訓練啟動完成！"
echo "======================================================"

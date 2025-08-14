#!/bin/bash

# TTT2 修復分支檔案清理腳本
# 實驗編號: CLEANUP_20250814
# 函式名稱: cleanup_unnecessary_files

echo "====================================================="
echo "TTT2 修復分支檔案清理腳本"
echo "實驗編號: CLEANUP_$(date +%Y%m%d%H%M)"
echo "====================================================="

# 創建備份目錄
BACKUP_DIR="backup_$(date +%Y%m%d%H%M)"
mkdir -p "$BACKUP_DIR"

echo "創建備份目錄: $BACKUP_DIR"

# 可以安全刪除的實驗分析工具
ANALYSIS_FILES=(
    
    "analyze_semantic_layer.py"
    "check_feature_shapes.py"
    "discrete_loss.py"
    "infer.py"
    "inspect_model.py"
    "loss_visualization.py"
    "tsne.py"
    "train.py"
    "try3.py"
    "time_alignment_fix.py"
    "update_min_content_samples.py"
    "verify_wavtokenizer.py"
)

# 外部測試相關檔案
EXTERNAL_TEST_FILES=(
    "test_ttt2_outside.py"
    "run_ttt2_outside_test.sh"
    "run_layer_visualization.sh"
    "analyze_audio_quality.py"
)

# 舊結果檔案
OLD_RESULT_FILES=(
    "result.png"
    "wavtokenizer.txt"
)

# 舊實驗目錄
OLD_DIRS=(
    "ttt2_outside_test_results"
    "b-0813"
)

echo "🗂️ 開始備份非必需檔案..."

# 備份分析工具檔案
echo "備份實驗分析工具..."
for file in "${ANALYSIS_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  備份: $file"
        mv "$file" "$BACKUP_DIR/"
    else
        echo "  不存在: $file"
    fi
done

# 備份外部測試檔案
echo "備份外部測試檔案..."
for file in "${EXTERNAL_TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  備份: $file"
        mv "$file" "$BACKUP_DIR/"
    else
        echo "  不存在: $file"
    fi
done

# 備份舊結果檔案
echo "備份舊結果檔案..."
for file in "${OLD_RESULT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  備份: $file"
        mv "$file" "$BACKUP_DIR/"
    else
        echo "  不存在: $file"
    fi
done

# 備份舊實驗目錄
echo "備份舊實驗目錄..."
for dir in "${OLD_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  備份目錄: $dir"
        mv "$dir" "$BACKUP_DIR/"
    else
        echo "  不存在目錄: $dir"
    fi
done

echo ""
echo "✅ 清理完成！"
echo ""
echo "📊 清理結果統計："
echo "備份目錄: $BACKUP_DIR"
echo "備份檔案數: $(find "$BACKUP_DIR" -type f | wc -l)"
echo "備份目錄數: $(find "$BACKUP_DIR" -type d | wc -l)"
echo ""

# 顯示剩餘的核心檔案
echo "🔧 剩餘核心檔案："
echo "主要執行檔案："
ls -la run_fixed_ttt2_branch.sh ttt2.py test_ttt2_fixes.py ttdata.py 2>/dev/null || echo "  部分核心檔案可能不存在"

echo ""
echo "核心目錄："
ls -ld decoder encoder config utils fairseq metrics 2>/dev/null || echo "  部分核心目錄可能不存在"

echo ""
echo "📝 配置和報告檔案："
ls -la *.yml *.txt REPORT.md 2>/dev/null || echo "  部分配置檔案可能不存在"

echo ""
echo "====================================================="
echo "🎯 TTT2 修復分支現在已清理完成"
echo "可以執行 ./run_fixed_ttt2_branch.sh 開始訓練"
echo "====================================================="

# 更新實驗報告
REPORT_FILE="REPORT.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "" >> $REPORT_FILE
echo "## 檔案清理作業 - CLEANUP_$(date +%Y%m%d%H%M)" >> $REPORT_FILE
echo "**執行時間:** $TIMESTAMP" >> $REPORT_FILE
echo "**函式名稱:** cleanup_unnecessary_files" >> $REPORT_FILE
echo "**備份目錄:** $BACKUP_DIR" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### 🗑️ 已清理檔案" >> $REPORT_FILE
echo "- **實驗分析工具:** ${#ANALYSIS_FILES[@]} 個檔案" >> $REPORT_FILE
echo "- **外部測試檔案:** ${#EXTERNAL_TEST_FILES[@]} 個檔案" >> $REPORT_FILE
echo "- **舊結果檔案:** ${#OLD_RESULT_FILES[@]} 個檔案" >> $REPORT_FILE
echo "- **舊實驗目錄:** ${#OLD_DIRS[@]} 個目錄" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "### ✅ 保留核心檔案" >> $REPORT_FILE
echo "- run_fixed_ttt2_branch.sh" >> $REPORT_FILE
echo "- ttt2.py" >> $REPORT_FILE
echo "- test_ttt2_fixes.py" >> $REPORT_FILE
echo "- ttdata.py" >> $REPORT_FILE
echo "- decoder/, encoder/, config/, utils/, fairseq/, metrics/ 目錄" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "**備註:** 所有清理的檔案已備份至 \`$BACKUP_DIR\` 目錄" >> $REPORT_FILE
echo "" >> $REPORT_FILE
echo "----" >> $REPORT_FILE

echo "已更新實驗報告: $REPORT_FILE"

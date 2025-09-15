#!/bin/bash
# 檔案清理腳本 - 安全地整理和移除不必要的測試檔案
# 執行前會先備份重要檔案

echo "🧹 開始檔案清理流程..."
echo "📅 時間: $(date)"

# 創建備份目錄
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📂 備份目錄: $BACKUP_DIR"

# 要清理的檔案類型
TEST_FILES=(
    "test_*.py"
    "simple_*.py" 
    "compare_*.py"
    "*.log"
)

# 要保留的重要檔案（不會被清理）
KEEP_FILES=(
    "test_fixed_training.py"  # 可能還需要用於驗證
    "test_output.log"         # 如果包含重要日誌
)

echo "🔍 分析要清理的檔案..."

# 顯示會被清理的檔案
echo "📋 發現以下測試檔案："
for pattern in "${TEST_FILES[@]}"; do
    for file in $pattern; do
        if [[ -f "$file" ]]; then
            # 檢查是否在保留列表中
            should_keep=false
            for keep_file in "${KEEP_FILES[@]}"; do
                if [[ "$file" == "$keep_file" ]]; then
                    should_keep=true
                    break
                fi
            done
            
            if [[ "$should_keep" == false ]]; then
                echo "  ❌ $file (會被清理)"
            else
                echo "  ✅ $file (會保留)"
            fi
        fi
    done
done

# 顯示會被移動到備份的檔案
echo ""
echo "📦 以下檔案會被移到備份目錄："
for pattern in "${TEST_FILES[@]}"; do
    for file in $pattern; do
        if [[ -f "$file" ]]; then
            should_keep=false
            for keep_file in "${KEEP_FILES[@]}"; do
                if [[ "$file" == "$keep_file" ]]; then
                    should_keep=true
                    break
                fi
            done
            
            if [[ "$should_keep" == false ]]; then
                echo "  📦 $file → $BACKUP_DIR/$file"
            fi
        fi
    done
done

echo ""
read -p "⚠️  確認要執行清理嗎？(y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 開始執行清理..."
    
    # 移動檔案到備份目錄
    moved_count=0
    for pattern in "${TEST_FILES[@]}"; do
        for file in $pattern; do
            if [[ -f "$file" ]]; then
                should_keep=false
                for keep_file in "${KEEP_FILES[@]}"; do
                    if [[ "$file" == "$keep_file" ]]; then
                        should_keep=true
                        break
                    fi
                done
                
                if [[ "$should_keep" == false ]]; then
                    mv "$file" "$BACKUP_DIR/"
                    echo "  ✅ 已移動: $file"
                    ((moved_count++))
                fi
            fi
        done
    done
    
    echo ""
    echo "✨ 清理完成！"
    echo "📊 統計:"
    echo "  - 移動檔案數: $moved_count"
    echo "  - 備份位置: $BACKUP_DIR"
    echo ""
    echo "💡 提示:"
    echo "  - 如需恢復檔案，請從 $BACKUP_DIR 複製回來"
    echo "  - 確認無需後可手動刪除備份目錄"
    echo "  - 使用 'rm -rf $BACKUP_DIR' 來徹底刪除備份"
    
else
    echo "❌ 取消清理操作"
    rm -rf "$BACKUP_DIR"
fi

echo ""
echo "🏁 清理流程結束"

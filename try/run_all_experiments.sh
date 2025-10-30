#!/bin/bash

# ========================================
# 批次執行所有驗證實驗
# ========================================
# 用途: 依序執行所有訓練速度驗證實驗
# 建議: 手動執行單個實驗，而非批次執行（每個需要數小時）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_SUMMARY="${SCRIPT_DIR}/../results/experiments_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "=========================================="
echo "訓練速度驗證實驗套件"
echo "=========================================="
echo ""
echo "⚠️  注意: 每個實驗需要數小時，建議手動執行單個實驗"
echo ""
echo "建議執行順序:"
echo "  1. bash try/experiment_3_mini_dataset.sh     (最快，~1-2小時)"
echo "  2. bash try/experiment_1_no_dropout.sh       (中等，~3-4小時)"
echo "  3. bash try/experiment_2_no_weight_decay.sh  (中等，~3-4小時)"
echo ""

read -p "確定要批次執行所有實驗嗎？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消批次執行"
    echo ""
    echo "💡 建議單獨執行實驗："
    echo "   bash try/experiment_3_mini_dataset.sh"
    exit 0
fi

echo ""
echo "開始批次執行實驗..."
echo ""

# 創建結果摘要檔案
cat > "$RESULTS_SUMMARY" << EOF
訓練速度驗證實驗 - 結果摘要
執行時間: $(date "+%Y-%m-%d %H:%M:%S")
========================================

EOF

# ========================================
# 實驗 3: 小資料集 (優先執行，最快)
# ========================================
echo "┌──────────────────────────────────────┐"
echo "│ 執行實驗 3: 小資料集 (14個音檔)      │"
echo "└──────────────────────────────────────┘"
echo ""

START_TIME=$(date +%s)
bash "${SCRIPT_DIR}/experiment_3_mini_dataset.sh"
EXP3_EXIT=$?
END_TIME=$(date +%s)
EXP3_DURATION=$((END_TIME - START_TIME))

cat >> "$RESULTS_SUMMARY" << EOF
實驗 3: 小資料集
  狀態: $([ $EXP3_EXIT -eq 0 ] && echo "✅ 成功" || echo "❌ 失敗 (exit code: $EXP3_EXIT)")
  耗時: $((EXP3_DURATION / 60)) 分鐘
  日誌: $(ls -t ../logs/exp3_mini_dataset_*.log 2>/dev/null | head -1)

EOF

echo ""
echo "實驗 3 完成 (耗時: $((EXP3_DURATION / 60)) 分鐘)"
echo ""

# ========================================
# 實驗 1: 移除 Dropout
# ========================================
echo "┌──────────────────────────────────────┐"
echo "│ 執行實驗 1: 移除 Dropout             │"
echo "└──────────────────────────────────────┘"
echo ""

START_TIME=$(date +%s)
bash "${SCRIPT_DIR}/experiment_1_no_dropout.sh"
EXP1_EXIT=$?
END_TIME=$(date +%s)
EXP1_DURATION=$((END_TIME - START_TIME))

cat >> "$RESULTS_SUMMARY" << EOF
實驗 1: 移除 Dropout
  狀態: $([ $EXP1_EXIT -eq 0 ] && echo "✅ 成功" || echo "❌ 失敗 (exit code: $EXP1_EXIT)")
  耗時: $((EXP1_DURATION / 60)) 分鐘
  日誌: $(ls -t ../logs/exp1_no_dropout_*.log 2>/dev/null | head -1)

EOF

echo ""
echo "實驗 1 完成 (耗時: $((EXP1_DURATION / 60)) 分鐘)"
echo ""

# ========================================
# 實驗 2: 移除 Weight Decay
# ========================================
echo "┌──────────────────────────────────────┐"
echo "│ 執行實驗 2: 移除 Weight Decay        │"
echo "└──────────────────────────────────────┘"
echo ""

START_TIME=$(date +%s)
bash "${SCRIPT_DIR}/experiment_2_no_weight_decay.sh"
EXP2_EXIT=$?
END_TIME=$(date +%s)
EXP2_DURATION=$((END_TIME - START_TIME))

cat >> "$RESULTS_SUMMARY" << EOF
實驗 2: 移除 Weight Decay
  狀態: $([ $EXP2_EXIT -eq 0 ] && echo "✅ 成功" || echo "❌ 失敗 (exit code: $EXP2_EXIT)")
  耗時: $((EXP2_DURATION / 60)) 分鐘
  日誌: $(ls -t ../logs/exp2_no_weight_decay_*.log 2>/dev/null | head -1)

EOF

echo ""
echo "實驗 2 完成 (耗時: $((EXP2_DURATION / 60)) 分鐘)"
echo ""

# ========================================
# 總結
# ========================================
TOTAL_DURATION=$((EXP1_DURATION + EXP2_DURATION + EXP3_DURATION))

cat >> "$RESULTS_SUMMARY" << EOF
========================================
總計耗時: $((TOTAL_DURATION / 3600)) 小時 $((TOTAL_DURATION % 3600 / 60)) 分鐘

快速分析命令:
  grep "Epoch 200" ../logs/exp1_*.log | grep "Train"
  grep "Epoch 200" ../logs/exp2_*.log | grep "Train"
  grep "Epoch 100" ../logs/exp3_*.log | grep "Train"

EOF

echo "=========================================="
echo "所有實驗已完成！"
echo "=========================================="
echo ""
echo "總耗時: $((TOTAL_DURATION / 3600)) 小時 $((TOTAL_DURATION % 3600 / 60)) 分鐘"
echo ""
echo "📊 結果摘要已保存到:"
echo "   $RESULTS_SUMMARY"
echo ""
echo "📁 查看結果:"
echo "   cat $RESULTS_SUMMARY"
echo ""
echo "🔍 詳細分析請參考:"
echo "   try/EXPERIMENTS_README.md"
echo ""

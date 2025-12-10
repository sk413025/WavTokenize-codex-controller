#!/bin/bash

# exp17 & exp18 實驗執行總結

echo "============================================================"
echo "exp17 & exp18 準備完成檢查清單"
echo "============================================================"
echo ""
export CUDA_VISIBLE_DEVICES=2
# 檢查文件是否存在
echo "📁 檢查必要文件..."
files=(
    "train_margin_loss.py"
    "train_curriculum.py"
    "run_exp17_margin.sh"
    "run_exp18_curriculum.sh"
    "exp17_18_core_functions.py"
    "EXP17_18_README.md"
    "QUICK_START_GUIDE.md"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
        all_exist=false
    fi
done

echo ""

# 檢查腳本執行權限
echo "🔐 檢查執行權限..."
for script in run_exp17_margin.sh run_exp18_curriculum.sh; do
    if [ -x "$script" ]; then
        echo "  ✅ $script"
    else
        echo "  ⚠️  $script (沒有執行權限，正在添加...)"
        chmod +x "$script"
        echo "  ✅ $script (已添加執行權限)"
    fi
done

echo ""

# 檢查梯度分析狀態
echo "📊 檢查梯度分析狀態..."
if [ -f "gradient_analysis.log" ]; then
    if pgrep -f "analyze_gradient_conflict.py" > /dev/null; then
        echo "  🔄 梯度分析正在執行中..."
        echo "     使用 'tail -f gradient_analysis.log' 監控進度"
    else
        last_line=$(tail -1 gradient_analysis.log)
        if echo "$last_line" | grep -q "完成\|完成！\|Completed"; then
            echo "  ✅ 梯度分析已完成"
            if [ -d "gradient_analysis" ]; then
                result_files=$(ls gradient_analysis/*.json 2>/dev/null | wc -l)
                if [ $result_files -gt 0 ]; then
                    echo "     找到 $result_files 個結果文件"
                    latest=$(ls -t gradient_analysis/*.json | head -1)
                    echo "     最新結果: $latest"
                fi
            fi
        else
            echo "  ❌ 梯度分析執行失敗或未完成"
            echo "     最後一行: $last_line"
            echo "     可以使用 'python analyze_gradient_conflict.py' 重新執行"
        fi
    fi
else
    echo "  ⚠️  尚未執行梯度分析"
    echo "     執行: python analyze_gradient_conflict.py"
fi

echo ""

# 檢查 GPU 狀態
echo "🖥️  檢查 GPU 狀態..."
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU 使用情況:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        gpu_id=$(echo $line | cut -d',' -f1)
        gpu_name=$(echo $line | cut -d',' -f2)
        mem_used=$(echo $line | cut -d',' -f3)
        mem_total=$(echo $line | cut -d',' -f4)
        mem_pct=$(awk "BEGIN {printf \"%.1f\", ($mem_used/$mem_total)*100}")
        echo "     GPU $gpu_id ($gpu_name): ${mem_used}MB / ${mem_total}MB (${mem_pct}%)"
    done
else
    echo "  ⚠️  nvidia-smi 不可用"
fi

echo ""
echo "============================================================"
echo "準備狀態總結"
echo "============================================================"

if $all_exist; then
    echo "✅ 所有必要文件已就緒"
else
    echo "❌ 有文件缺失，請檢查上方列表"
fi

echo ""
echo "📋 下一步操作建議："
echo ""
echo "1. 查看梯度分析結果（如果已完成）："
echo "   cat gradient_analysis/gradient_analysis_*.json | jq '.mean_cosine, .conflict_ratio'"
echo ""
echo "2. 執行 exp17 (Margin Loss)："
echo "   ./run_exp17_margin.sh &"
echo "   tail -f exp17.log"
echo ""
echo "3. 執行 exp18 (Curriculum Learning)："
echo "   ./run_exp18_curriculum.sh &"
echo "   tail -f exp18.log"
echo ""
echo "4. 同時執行兩個實驗（使用不同 GPU）："
echo "   CUDA_VISIBLE_DEVICES=0 ./run_exp17_margin.sh > exp17.log 2>&1 &"
echo "   CUDA_VISIBLE_DEVICES=1 ./run_exp18_curriculum.sh > exp18.log 2>&1 &"
echo ""
echo "5. 監控訓練進度："
echo "   watch -n 5 'tail -20 exp17.log exp18.log'"
echo ""

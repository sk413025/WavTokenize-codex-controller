#!/bin/bash

# WavTokenizer 特徵分析執行腳本
# 實驗編號: EXP09_SHELL
# 日期: 2025-08-12
# 作者: GitHub Copilot

# 獲取腳本所在目錄的絕對路徑
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 檢查特徵形狀並執行分析
run_analysis() {
    echo "執行特徵形狀檢查和分析..."
    python ${SCRIPT_DIR}/run_feature_analysis.py --task "檢查特徵形狀"
}

# 分析指定的t-SNE輸出目錄
analyze_tsne_output() {
    if [ -z "$1" ]; then
        echo "錯誤: 請提供t-SNE輸出目錄路徑"
        exit 1
    fi
    
    TSNE_DIR=$1
    EXP_ID=${2:-"EXP_TSNE_$(date +%Y%m%d_%H%M%S)"}
    
    echo "分析t-SNE輸出目錄: ${TSNE_DIR}"
    echo "實驗ID: ${EXP_ID}"
    
    python ${SCRIPT_DIR}/run_feature_analysis.py --tsne-output-dir "${TSNE_DIR}" --experiment-id "${EXP_ID}"
}

# 分析指定的特徵檔案
analyze_features() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "錯誤: 請提供增強特徵檔案路徑和目標特徵檔案路徑"
        exit 1
    fi
    
    ENHANCED_FEATURES=$1
    TARGET_FEATURES=$2
    OUTPUT_DIR=${3:-"results/feature_analysis_$(date +%Y%m%d_%H%M%S)"}
    EXP_ID=${4:-"EXP_FEAT_$(date +%Y%m%d_%H%M%S)"}
    
    echo "分析特徵檔案:"
    echo "增強特徵: ${ENHANCED_FEATURES}"
    echo "目標特徵: ${TARGET_FEATURES}"
    echo "輸出目錄: ${OUTPUT_DIR}"
    echo "實驗ID: ${EXP_ID}"
    
    python ${SCRIPT_DIR}/run_feature_analysis.py --enhanced-features "${ENHANCED_FEATURES}" --target-features "${TARGET_FEATURES}" --output-dir "${OUTPUT_DIR}" --experiment-id "${EXP_ID}"
}

# 使用幫助
show_help() {
    echo "使用方法: $(basename $0) [選項]"
    echo ""
    echo "選項:"
    echo "  --run-analysis              檢查特徵形狀並執行基本分析"
    echo "  --analyze-tsne <目錄>       分析指定的t-SNE輸出目錄"
    echo "  --analyze-features <增強特徵> <目標特徵> [輸出目錄] [實驗ID]"
    echo "                              分析指定的特徵檔案"
    echo "  --help                      顯示此幫助訊息"
    echo ""
    echo "範例:"
    echo "  $(basename $0) --run-analysis"
    echo "  $(basename $0) --analyze-tsne results/tsne_outputs/output2"
    echo "  $(basename $0) --analyze-features path/to/enhanced.pt path/to/target.pt"
}

# 主函數
main() {
    # 檢查命令行參數
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # 解析命令行參數
    case "$1" in
        --run-analysis)
            run_analysis
            ;;
        --analyze-tsne)
            analyze_tsne_output "$2" "$3"
            ;;
        --analyze-features)
            analyze_features "$2" "$3" "$4" "$5"
            ;;
        --help)
            show_help
            ;;
        *)
            echo "未知選項: $1"
            show_help
            exit 1
            ;;
    esac
}

# 執行主函數
main "$@"

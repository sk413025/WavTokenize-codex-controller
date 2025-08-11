#!/bin/bash
# 離散編碼分析與增強實驗批處理腳本
# 實驗編號: EXP08
# 日期: 2025-08-01
# 作者: 實驗腳本生成器

# 設置基本參數
CLEAN_DIR="/home/sbplab/ruizi/WavTokenize/1c"
NOISY_DIR="/home/sbplab/ruizi/WavTokenize/1b"
OUTPUT_BASE_DIR="/home/sbplab/ruizi/WavTokenize/results/discrete_experiments"
GPU_ID=0

# 建立輸出目錄
mkdir -p "${OUTPUT_BASE_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="${OUTPUT_BASE_DIR}/exp_${TIMESTAMP}"
mkdir -p "${EXP_DIR}"

# 日誌文件
LOG_FILE="${EXP_DIR}/experiment_${TIMESTAMP}.log"

# 函數：記錄訊息到日誌
log_message() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "${LOG_FILE}"
}

log_message "開始離散編碼分析與增強實驗 (${TIMESTAMP})"
log_message "乾淨音頻目錄: ${CLEAN_DIR}"
log_message "帶噪音頻目錄: ${NOISY_DIR}"
log_message "輸出目錄: ${EXP_DIR}"
log_message "使用 GPU: ${GPU_ID}"

# 步驟 1: 基礎離散編碼分析
log_message "步驟 1: 執行離散編碼基礎分析"
python exp_discrete_analysis.py \
    --clean_dir "${CLEAN_DIR}" \
    --noisy_dir "${NOISY_DIR}" \
    --output "${EXP_DIR}/analysis" \
    --max_samples 20 \
    --gpu "${GPU_ID}" | tee -a "${LOG_FILE}"

# 步驟 2: 層交換實驗
log_message "步驟 2: 執行層交換實驗"
python exp_layer_swap.py \
    --clean_dir "${CLEAN_DIR}" \
    --noisy_dir "${NOISY_DIR}" \
    --output "${EXP_DIR}/layer_swap" \
    --n_samples 10 \
    --gpu "${GPU_ID}" | tee -a "${LOG_FILE}"

# 步驟 3: 特徵可視化
log_message "步驟 3: 執行特徵可視化"
python exp_visualize_discrete.py \
    --output "${EXP_DIR}/visualization" \
    --gpu "${GPU_ID}" | tee -a "${LOG_FILE}"

# 步驟 4: 離散編碼增強模型訓練
# 創建配置文件
CONFIG_FILE="${EXP_DIR}/training_config.yaml"
cat > "${CONFIG_FILE}" << EOF
# 離散編碼增強模型訓練配置
# 實驗日期: $(date +"%Y-%m-%d")

n_layers: 8
codebook_size: 4096
hidden_dim: 512
content_layers: [0, 1, 2]
speaker_layers: [3, 4]
batch_size: 16
epochs: 50
learning_rate: 0.0001
content_weight: 0.8
speaker_weight: 0.2
use_mel_loss: true
max_samples: 100
EOF

log_message "步驟 4: 執行離散編碼增強模型訓練"
python exp_discrete_training.py \
    --clean_dir "${CLEAN_DIR}" \
    --noisy_dir "${NOISY_DIR}" \
    --output "${EXP_DIR}/training" \
    --config "${CONFIG_FILE}" \
    --gpu "${GPU_ID}" | tee -a "${LOG_FILE}"

# 步驟 5: 實驗結果分析與總結
log_message "步驟 5: 生成實驗結果報告"
REPORT_CONTENT=$(cat << EOF
# 離散編碼分析與增強實驗報告
實驗ID: EXP_${TIMESTAMP}
日期: $(date +"%Y-%m-%d")

## 實驗概述
本實驗分析了 WavTokenizer 模型的離散編碼特性，並實現了一種基於離散特徵的語音增強方法。
實驗包含離散編碼分析、層交換實驗、特徵可視化和增強模型訓練等步驟。

## 實驗結果概要
- 分析目錄: ${EXP_DIR}/analysis
- 層交換結果: ${EXP_DIR}/layer_swap
- 可視化結果: ${EXP_DIR}/visualization
- 訓練結果: ${EXP_DIR}/training

## 後續工作
1. 優化模型結構以提高性能
2. 嘗試不同的層功能分配策略
3. 探索更多的損失函數組合
4. 在更大的數據集上進行驗證

EOF
)

# 保存實驗報告
REPORT_FILE="${EXP_DIR}/experiment_report.md"
echo "${REPORT_CONTENT}" > "${REPORT_FILE}"
log_message "實驗報告已保存至: ${REPORT_FILE}"

# 更新中央報告文件
python exp_update_report.py \
    --exp_id "EXP_${TIMESTAMP}" \
    --exp_name "離散編碼分析與增強實驗" \
    --config_file "${CONFIG_FILE}" \
    --results_dir "${EXP_DIR}" \
    --description "分析 WavTokenizer 離散編碼的特性，並實現基於離散特徵的語音增強方法，針對不同層實施不同的處理策略。" \
    --report_file "REPORT.md" \
    --exp_date "$(date +"%Y-%m-%d")" | tee -a "${LOG_FILE}"

log_message "實驗完成! 結果保存在: ${EXP_DIR}"

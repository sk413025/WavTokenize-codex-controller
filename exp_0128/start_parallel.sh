#!/bin/bash

# ============================================================
# exp_0128: 平行執行兩個實驗
# - GPU 0: 實驗 2 (Noise-Balanced Sampling)
# - GPU 1: 實驗 1 (TracIn-Weighted Soft Reweighting)
# ============================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="exp_0128/logs_parallel_${TIMESTAMP}"
mkdir -p ${LOG_DIR}

echo "============================================================"
echo "Starting Parallel Experiments (exp_0128)"
echo "============================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "GPU 0: Experiment 2 (Noise-Balanced Sampling)"
echo "GPU 1: Experiment 1 (TracIn-Weighted Soft Reweighting)"
echo "Log directory: ${LOG_DIR}"
echo "============================================================"

# 啟動實驗 2 (GPU 0) - 背景執行
echo ""
echo "Starting Experiment 2 on GPU 0..."
nohup bash exp_0128/noise_balanced_sampling/run_exp2.sh > ${LOG_DIR}/exp2_gpu0.log 2>&1 &
EXP2_PID=$!
echo "  PID: ${EXP2_PID}"
echo "  Log: ${LOG_DIR}/exp2_gpu0.log"

# 等待 2 秒確保第一個實驗啟動
sleep 2

# 啟動實驗 1 (GPU 1) - 背景執行
echo ""
echo "Starting Experiment 1 on GPU 1..."
nohup bash exp_0128/soft_reweighting/run_exp1.sh > ${LOG_DIR}/exp1_gpu1.log 2>&1 &
EXP1_PID=$!
echo "  PID: ${EXP1_PID}"
echo "  Log: ${LOG_DIR}/exp1_gpu1.log"

# 保存 PID 到文件
echo ${EXP2_PID} > ${LOG_DIR}/exp2_pid.txt
echo ${EXP1_PID} > ${LOG_DIR}/exp1_pid.txt

echo ""
echo "============================================================"
echo "Both experiments started in background!"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  Experiment 2: tail -f ${LOG_DIR}/exp2_gpu0.log"
echo "  Experiment 1: tail -f ${LOG_DIR}/exp1_gpu1.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Check running processes:"
echo "  ps -p ${EXP2_PID},${EXP1_PID} -o pid,cmd"
echo ""
echo "Stop experiments (if needed):"
echo "  kill ${EXP2_PID}  # Stop Experiment 2"
echo "  kill ${EXP1_PID}  # Stop Experiment 1"
echo "============================================================"

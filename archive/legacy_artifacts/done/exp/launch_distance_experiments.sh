#!/bin/bash
#
# 啟動 VQ Distance 實驗
#
# 實驗組別:
#   1. Baseline (CE only) - 不使用 distances
#   2. Exp-1: Soft Target (α=0.5) - 平衡 soft/hard
#   3. Exp-2: Soft Target (α=0.7) - 更重視 soft target
#   4. Exp-3: Hybrid Loss (0.3/0.3/0.4) - 三種 loss 混合
#
# 使用方式:
#   # 運行所有實驗（並行前3個）
#   bash launch_distance_experiments.sh all
#
#   # 運行單個實驗
#   bash launch_distance_experiments.sh baseline
#   bash launch_distance_experiments.sh exp1
#   bash launch_distance_experiments.sh exp2
#   bash launch_distance_experiments.sh exp3
#

set -e  # 遇到錯誤立即退出

# ============================================================
# 配置區
# ============================================================

# 通用配置
CACHE_DIR="./data_with_distances"
OUTPUT_DIR="./distance_experiments"
BATCH_SIZE=28
NUM_WORKERS=4
NUM_EPOCHS=200

# 模型配置
D_MODEL=512
NHEAD=8
NUM_LAYERS=4
DIM_FEEDFORWARD=2048
DROPOUT=0.1
FUSION_METHOD="add"

# 訓練配置
LEARNING_RATE=1e-3  # 提高 10 倍以加快學習
WEIGHT_DECAY=0.01
GRADIENT_CLIP=1.0

# Scheduler 配置
USE_SCHEDULER="--use_scheduler"
SCHEDULER_PATIENCE=10
SCHEDULER_FACTOR=0.5

# Early Stopping
EARLY_STOPPING_PATIENCE=30

# Checkpoint
SAVE_CHECKPOINT_FREQ=50

# Warm-up 配置
USE_WARMUP="--use_warmup"
WARMUP_EPOCHS=5  # 縮短至 5 epochs 防止 early collapse

# 防止 Model Collapse 配置
USE_CLASS_WEIGHTS="--use_class_weights"  # 降低 Token 453 權重
MAJORITY_WEIGHT=0.01  # Token 453 的權重 (1% of normal)
ENTROPY_WEIGHT=0.01  # 熵正則化權重

# WavTokenizer 配置
WAVTOKENIZER_CONFIG="../../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOKENIZER_CHECKPOINT="../../models/wavtokenizer_large_speech_320_24k.ckpt"

# GPU 配置（自動檢測可用 GPU）
AVAILABLE_GPUS=(0 1 2)

# 隨機種子
SEED=42

# ============================================================
# 實驗定義
# ============================================================

function run_baseline() {
    local GPU=$1
    local EXP_NAME="baseline"
    
    echo "=========================================="
    echo "啟動 Baseline (CE only) on GPU ${GPU}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU} python train_with_distances.py \
        --exp_name ${EXP_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --cache_dir ${CACHE_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --loss_type ce \
        ${USE_CLASS_WEIGHTS} \
        --majority_weight ${MAJORITY_WEIGHT} \
        --entropy_weight ${ENTROPY_WEIGHT} \
        --d_model ${D_MODEL} \
        --nhead ${NHEAD} \
        --num_layers ${NUM_LAYERS} \
        --dim_feedforward ${DIM_FEEDFORWARD} \
        --dropout ${DROPOUT} \
        --fusion_method ${FUSION_METHOD} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --gradient_clip ${GRADIENT_CLIP} \
        ${USE_SCHEDULER} \
        --scheduler_patience ${SCHEDULER_PATIENCE} \
        --scheduler_factor ${SCHEDULER_FACTOR} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --save_checkpoint_freq ${SAVE_CHECKPOINT_FREQ} \
        --wavtokenizer_config ${WAVTOKENIZER_CONFIG} \
        --wavtokenizer_checkpoint ${WAVTOKENIZER_CHECKPOINT} \
        --seed ${SEED} \
        > ${OUTPUT_DIR}/${EXP_NAME}_training.log 2>&1
    
    echo "✅ Baseline 完成"
}

function run_exp1() {
    local GPU=$1
    local EXP_NAME="exp1_soft_05"
    
    echo "=========================================="
    echo "啟動 Exp-1: Soft Target (α=0.5) on GPU ${GPU}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU} python train_with_distances.py \
        --exp_name ${EXP_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --cache_dir ${CACHE_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --loss_type soft \
        --temperature 2.0 \
        --alpha 0.5 \
        ${USE_CLASS_WEIGHTS} \
        --majority_weight ${MAJORITY_WEIGHT} \
        --entropy_weight ${ENTROPY_WEIGHT} \
        ${USE_WARMUP} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --d_model ${D_MODEL} \
        --nhead ${NHEAD} \
        --num_layers ${NUM_LAYERS} \
        --dim_feedforward ${DIM_FEEDFORWARD} \
        --dropout ${DROPOUT} \
        --fusion_method ${FUSION_METHOD} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --gradient_clip ${GRADIENT_CLIP} \
        ${USE_SCHEDULER} \
        --scheduler_patience ${SCHEDULER_PATIENCE} \
        --scheduler_factor ${SCHEDULER_FACTOR} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --save_checkpoint_freq ${SAVE_CHECKPOINT_FREQ} \
        --wavtokenizer_config ${WAVTOKENIZER_CONFIG} \
        --wavtokenizer_checkpoint ${WAVTOKENIZER_CHECKPOINT} \
        --seed ${SEED} \
        > ${OUTPUT_DIR}/${EXP_NAME}_training.log 2>&1
    
    echo "✅ Exp-1 完成"
}

function run_exp2() {
    local GPU=$1
    local EXP_NAME="exp2_soft_07"
    
    echo "=========================================="
    echo "啟動 Exp-2: Soft Target (α=0.7) on GPU ${GPU}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU} python train_with_distances.py \
        --exp_name ${EXP_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --cache_dir ${CACHE_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --loss_type soft \
        --temperature 2.0 \
        --alpha 0.7 \
        ${USE_CLASS_WEIGHTS} \
        --majority_weight ${MAJORITY_WEIGHT} \
        --entropy_weight ${ENTROPY_WEIGHT} \
        ${USE_WARMUP} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --d_model ${D_MODEL} \
        --nhead ${NHEAD} \
        --num_layers ${NUM_LAYERS} \
        --dim_feedforward ${DIM_FEEDFORWARD} \
        --dropout ${DROPOUT} \
        --fusion_method ${FUSION_METHOD} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --gradient_clip ${GRADIENT_CLIP} \
        ${USE_SCHEDULER} \
        --scheduler_patience ${SCHEDULER_PATIENCE} \
        --scheduler_factor ${SCHEDULER_FACTOR} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --save_checkpoint_freq ${SAVE_CHECKPOINT_FREQ} \
        --wavtokenizer_config ${WAVTOKENIZER_CONFIG} \
        --wavtokenizer_checkpoint ${WAVTOKENIZER_CHECKPOINT} \
        --seed ${SEED} \
        > ${OUTPUT_DIR}/${EXP_NAME}_training.log 2>&1
    
    echo "✅ Exp-2 完成"
}

function run_exp3() {
    local GPU=$1
    local EXP_NAME="exp3_hybrid"
    
    echo "=========================================="
    echo "啟動 Exp-3: Hybrid Loss (0.3/0.3/0.4) on GPU ${GPU}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU} python train_with_distances.py \
        --exp_name ${EXP_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --cache_dir ${CACHE_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --loss_type hybrid \
        --temperature 2.0 \
        --alpha 0.3 \
        --beta 0.3 \
        --gamma 0.4 \
        ${USE_CLASS_WEIGHTS} \
        --majority_weight ${MAJORITY_WEIGHT} \
        --entropy_weight ${ENTROPY_WEIGHT} \
        ${USE_WARMUP} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --d_model ${D_MODEL} \
        --nhead ${NHEAD} \
        --num_layers ${NUM_LAYERS} \
        --dim_feedforward ${DIM_FEEDFORWARD} \
        --dropout ${DROPOUT} \
        --fusion_method ${FUSION_METHOD} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --gradient_clip ${GRADIENT_CLIP} \
        ${USE_SCHEDULER} \
        --scheduler_patience ${SCHEDULER_PATIENCE} \
        --scheduler_factor ${SCHEDULER_FACTOR} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --save_checkpoint_freq ${SAVE_CHECKPOINT_FREQ} \
        --wavtokenizer_config ${WAVTOKENIZER_CONFIG} \
        --wavtokenizer_checkpoint ${WAVTOKENIZER_CHECKPOINT} \
        --seed ${SEED} \
        > ${OUTPUT_DIR}/${EXP_NAME}_training.log 2>&1
    
    echo "✅ Exp-3 完成"
}

# ============================================================
# 主控制邏輯
# ============================================================

function show_usage() {
    echo "使用方式:"
    echo "  bash $0 all          # 運行所有實驗（並行前3個）"
    echo "  bash $0 baseline     # 只運行 Baseline"
    echo "  bash $0 exp1         # 只運行 Exp-1 (Soft α=0.5)"
    echo "  bash $0 exp2         # 只運行 Exp-2 (Soft α=0.7)"
    echo "  bash $0 exp3         # 只運行 Exp-3 (Hybrid)"
    exit 1
}

function check_prerequisites() {
    echo "檢查前置條件..."
    
    # 檢查數據目錄
    if [ ! -d "${CACHE_DIR}" ]; then
        echo "❌ 錯誤: 數據目錄不存在: ${CACHE_DIR}"
        exit 1
    fi
    
    if [ ! -f "${CACHE_DIR}/cache_with_distances.h5" ]; then
        echo "❌ 錯誤: HDF5 數據不存在: ${CACHE_DIR}/cache_with_distances.h5"
        exit 1
    fi
    
    # 檢查 WavTokenizer
    if [ ! -f "${WAVTOKENIZER_CONFIG}" ]; then
        echo "❌ 錯誤: WavTokenizer 配置不存在: ${WAVTOKENIZER_CONFIG}"
        exit 1
    fi
    
    if [ ! -f "${WAVTOKENIZER_CHECKPOINT}" ]; then
        echo "❌ 錯誤: WavTokenizer checkpoint 不存在: ${WAVTOKENIZER_CHECKPOINT}"
        exit 1
    fi
    
    # 檢查 GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ 錯誤: nvidia-smi 不可用"
        exit 1
    fi
    
    # 創建輸出目錄
    mkdir -p ${OUTPUT_DIR}
    
    echo "✅ 前置條件檢查通過"
    echo ""
}

function run_all() {
    echo "=========================================="
    echo "運行所有實驗"
    echo "=========================================="
    echo "GPU 0: Baseline"
    echo "GPU 1: Exp-1 (Soft α=0.5)"
    echo "GPU 2: Exp-2 (Soft α=0.7)"
    echo "等待完成後運行:"
    echo "GPU 1: Exp-3 (Hybrid)"
    echo ""
    
    # 並行運行前 3 個實驗
    echo "啟動並行實驗 (Baseline, Exp-1, Exp-2)..."
    run_baseline ${AVAILABLE_GPUS[0]} &
    PID_BASELINE=$!
    
    run_exp1 ${AVAILABLE_GPUS[1]} &
    PID_EXP1=$!
    
    run_exp2 ${AVAILABLE_GPUS[2]} &
    PID_EXP2=$!
    
    # 等待並行實驗完成
    echo "等待並行實驗完成..."
    echo "  - Baseline (PID: $PID_BASELINE)"
    echo "  - Exp-1 (PID: $PID_EXP1)"
    echo "  - Exp-2 (PID: $PID_EXP2)"
    echo ""
    
    wait $PID_BASELINE
    echo "✅ Baseline 完成"
    
    wait $PID_EXP1
    echo "✅ Exp-1 完成"
    
    wait $PID_EXP2
    echo "✅ Exp-2 完成"
    
    echo ""
    echo "並行實驗全部完成，開始運行 Exp-3..."
    
    # 運行 Exp-3
    run_exp3 ${AVAILABLE_GPUS[1]}
    
    echo ""
    echo "=========================================="
    echo "✅ 所有實驗完成！"
    echo "=========================================="
    echo "結果目錄: ${OUTPUT_DIR}"
    echo ""
    echo "查看結果:"
    echo "  - Baseline:      ${OUTPUT_DIR}/baseline/logs/training.log"
    echo "  - Exp-1 (α=0.5): ${OUTPUT_DIR}/exp1_soft_05/logs/training.log"
    echo "  - Exp-2 (α=0.7): ${OUTPUT_DIR}/exp2_soft_07/logs/training.log"
    echo "  - Exp-3 (Hybrid): ${OUTPUT_DIR}/exp3_hybrid/logs/training.log"
    echo ""
}

# ============================================================
# 主程序入口
# ============================================================

if [ $# -eq 0 ]; then
    show_usage
fi

check_prerequisites

COMMAND=$1

case ${COMMAND} in
    all)
        run_all
        ;;
    baseline)
        run_baseline ${AVAILABLE_GPUS[0]}
        ;;
    exp1)
        run_exp1 ${AVAILABLE_GPUS[1]}
        ;;
    exp2)
        run_exp2 ${AVAILABLE_GPUS[2]}
        ;;
    exp3)
        run_exp3 ${AVAILABLE_GPUS[1]}
        ;;
    *)
        echo "❌ 未知命令: ${COMMAND}"
        show_usage
        ;;
esac

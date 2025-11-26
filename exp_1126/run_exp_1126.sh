#!/bin/bash

# Exp_1126 Training Script - 1D Wasserstein Distance Loss with SCALING
# 使用 Loss Scaling 使 Wasserstein 與 CE 同量級
# 重現 commit: 0502ca619f86f1d336eba0eb23e507b46207eca5

set -e

# 使用 GPU 0 (可根據需要修改)
export CUDA_VISIBLE_DEVICES=0

# Configuration - 數據路徑
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_CACHE="/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt"
VAL_CACHE="/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt"
WAVTOK_CONFIG="/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT="/home/sbplab/ruizi/c_code/models/wavtokenizer_large_speech_320_24k.ckpt"

# Training hyperparameters (1D Wasserstein 可以用更大 batch size!)
BATCH_SIZE=28  # 1D 版本內存需求低，可增加 batch size
EPOCHS=120
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
DROPOUT=0.15
PATIENCE=10

# Wasserstein Loss hyperparameters - 1D VERSION with SCALING
WASSERSTEIN_ALPHA=1.0    # 100% Wasserstein (因為已經縮放，可以使用純 Wasserstein)
SCALE_FACTOR=23.0        # 縮放因子 (根據 analyze_loss_scale.py 結果)

# Model hyperparameters
D_MODEL=512
NHEAD=8
NUM_LAYERS=4
DIM_FFN=2048
FUSION_METHOD="cross_attn"

echo "========================================================================"
echo "Exp_1126: 1D Wasserstein Loss with Loss Scaling (α=1.0)"
echo "重現 commit: 0502ca619f86f1d336eba0eb23e507b46207eca5"
echo "========================================================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Wasserstein Alpha: $WASSERSTEIN_ALPHA (100% Wasserstein)"
echo "Scale Factor: ${SCALE_FACTOR}x (使 Wasserstein 與 CE 同量級)"
echo "Version: 1D (Memory-Efficient)"
echo "Batch Size: $BATCH_SIZE"
echo ""
echo "✓ Loss Scaling 優勢:"
echo "  - Wasserstein Loss 縮放至與 CE Loss 同量級"
echo "  - 純 Wasserstein (α=1.0) 現在有足夠大的 gradient"
echo "  - 學習速度與 CE 相當，同時保留 token 距離資訊"
echo ""

# 切換到腳本目錄
cd "$SCRIPT_DIR"

# Run training
python train_exp_1126.py \
    --train_cache ${TRAIN_CACHE} \
    --val_cache ${VAL_CACHE} \
    --wavtok_config ${WAVTOK_CONFIG} \
    --wavtok_ckpt ${WAVTOK_CKPT} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --dropout ${DROPOUT} \
    --patience ${PATIENCE} \
    --d_model ${D_MODEL} \
    --nhead ${NHEAD} \
    --num_layers ${NUM_LAYERS} \
    --dim_feedforward ${DIM_FFN} \
    --fusion_method ${FUSION_METHOD} \
    --use_learnable_gate \
    --scheduler plateau \
    --num_workers 4 \
    --save_interval 20 \
    --num_vis_samples 3 \
    --device cuda \
    --wasserstein_alpha ${WASSERSTEIN_ALPHA} \
    --wasserstein_scale_factor ${SCALE_FACTOR} \
    --use_1d_wasserstein

echo ""
echo "========================================================================"
echo "Training completed!"
echo "========================================================================"

#!/bin/bash
# exp_1207: 純 Feature Loss 實驗
#
# 假設：如果 z_noisy ≈ z_clean，則 token_noisy == token_clean
# 測試：只用 Feature MSE Loss，不用任何 Distance Loss
# 使用 GPU 1 (RTX 2080 Ti, 11GB)
export CUDA_VISIBLE_DEVICES=1
# 優化 CUDA 記憶體分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "exp_1207: 純 Feature Loss 實驗"
echo "============================================================"
echo ""
echo "假設: Feature Loss 讓 z_noisy → z_clean"
echo "      則 argmin(z_noisy) == argmin(z_clean)"
echo ""

# ============================================================
# 實驗: 標準配置
# ============================================================
echo "============================================================"
echo "實驗: feature_only (標準配置, 固定學習率)"
echo "============================================================"

python train.py \
    --exp_name feature_only \
    --num_epochs 30 \
    --batch_size 28 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --save_interval 10 \
    --plot_interval 10 \
    --audio_interval 10

echo ""
echo "實驗完成！"
echo ""

# ============================================================
# 總結
# ============================================================
echo "============================================================"
echo "實驗完成！"
echo "============================================================"
echo ""
echo "結果位置:"
echo "  - experiments/feature_only/"
echo ""
echo "分析方式:"
echo "  python -c \"import json; h=json.load(open('experiments/feature_only/training_history.json')); print('Epoch 1:', h['train_token_acc'][0]*100, '%'); print('Final:', h['train_token_acc'][-1]*100, '%')\""

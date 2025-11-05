#!/bin/bash

# Cross-Attention Fusion 實驗啟動腳本
# 實驗編號: EXP-20251105-CrossAttn
# 目的: 驗證假設 2 - Speaker Embedding 影響力不足

echo "================================================================================"
echo "Cross-Attention Speaker Fusion Experiment"
echo "================================================================================"
echo "實驗編號: EXP-20251105-CrossAttn"
echo "時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "關鍵設定:"
echo "  - Fusion 方式: Cross-Attention (替代 Additive)"
echo "  - Batch size: 64 (大幅提升)"
echo "  - Learning rate: 1e-4 (固定)"
echo "  - Weight decay: 0.0 (無)"
echo "  - Scheduler: None (無)"
echo "  - Epochs: 100"
echo ""
echo "預期改善:"
echo "  - Speaker Influence: <5% → >20%"
echo "  - Val Accuracy: 38% → 43-47%"
echo "  - Token 0 預測率: 32% → 20-25%"
echo "================================================================================"
echo ""

# 確認緩存存在
if [ ! -f "./data/train_cache.pt" ]; then
    echo "❌ 錯誤: 訓練集緩存不存在 (./data/train_cache.pt)"
    echo "請先運行: python preprocess_zeroshot_cache.py"
    exit 1
fi

if [ ! -f "./data/val_cache.pt" ]; then
    echo "❌ 錯誤: 驗證集緩存不存在 (./data/val_cache.pt)"
    echo "請先運行: python preprocess_zeroshot_cache.py"
    exit 1
fi

echo "✓ 緩存檢查通過"
echo ""

# 設定 GPU (使用 PCI_BUS_ID 確保編號一致)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

echo "使用 GPU: $CUDA_VISIBLE_DEVICES (PCI Bus Order)"
echo ""

# 運行訓練
echo "開始訓練..."
echo ""

python -u train_crossattn_cached.py \
    --cache_dir ./data \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --dropout 0.1 \
    --num_workers 4 \
    2>&1 | tee crossattn_training.log

echo ""
echo "================================================================================"
echo "訓練完成！"
echo "================================================================================"
echo "查看結果:"
echo "  - 訓練日誌: results/crossattn_100epochs_*/training.log"
echo "  - 損失曲線: results/crossattn_100epochs_*/loss_curves_final.png"
echo "  - 最佳模型: results/crossattn_100epochs_*/best_model.pth"
echo "================================================================================"

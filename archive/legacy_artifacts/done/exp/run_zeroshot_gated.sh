#!/bin/bash
#
# Zero-Shot Denoising 實驗：Gated Fusion
#
# 改進：使用 Gated Fusion 替代 Simple Addition
#
# 預期效果：
# - 驗證準確率: 39.29% → 40.5-41.5%
# - 更好地利用 speaker 信息
#

export PYTHONPATH="/home/sbplab/ruizi/c_code:${PYTHONPATH}"

cd /home/sbplab/ruizi/c_code/done/exp

# 確認緩存存在
if [ ! -f "./data/train_cache.pt" ]; then
    echo "錯誤: 找不到 train_cache.pt，請先運行 preprocess_zeroshot_cache.py"
    exit 1
fi

# 創建輸出目錄（帶時間戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results/zeroshot_gated_${TIMESTAMP}"

echo "=========================================="
echo "Zero-Shot Denoising: Gated Fusion"
echo "=========================================="
echo "實驗配置:"
echo "  - num_layers: 4 (保持不變)"
echo "  - fusion: Gated Fusion ⭐ (從 Addition 改進)"
echo "  - dropout: 0.2"
echo "  - batch_size: 28"
echo "  - learning_rate: 1e-4"
echo ""
echo "模型改進:"
echo "  - 總參數: ~14.8M (基本相同)"
echo "  - Gated Fusion 額外參數: 0.53M"
echo "  - 動態學習 token vs speaker 的融合權重"
echo ""
echo "預期效果:"
echo "  - 訓練速度: ~2.5 小時 (略慢於原版)"
echo "  - 驗證準確率: 40.5-41.5% (提升 1.2-2.2%)"
echo "  - 更好的 speaker conditioning"
echo ""
echo "比較基準:"
echo "  - Baseline: 38.19%"
echo "  - Simple Addition (layer=4): 39.29% (+1.10%)"
echo "  - num_layers=3: 38.69% (+0.50%, 變差)"
echo "  - 目標: ≥ 40.5% (+2.31%)"
echo ""
echo "輸出目錄: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# 訓練
python train_zeroshot_gated_cached.py \
    --cache_dir ./data \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs 100 \
    --batch_size 28 \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --d_model 512 \
    --nhead 8 \
    --num_layers 4 \
    --dim_feedforward 2048 \
    --dropout 0.2

echo ""
echo "=========================================="
echo "訓練完成！"
echo "=========================================="
echo "結果位置: ${OUTPUT_DIR}"
echo ""
echo "查看結果:"
echo "  - 訓練日誌: ${OUTPUT_DIR}/training.log"
echo "  - Loss 曲線: ${OUTPUT_DIR}/loss_curves_epoch_50.png"
echo "  - 音頻樣本: ${OUTPUT_DIR}/audio_samples/epoch_50/"
echo "  - 最佳模型: ${OUTPUT_DIR}/best_model.pth"
echo ""
echo "評估標準:"
echo "  - 若 Val Acc ≥ 40.5%: ✅ Gated Fusion 有效"
echo "  - 若 Val Acc 39.5-40.5%: ⚠️  略有提升"
echo "  - 若 Val Acc < 39.5%: ❌ Gated Fusion 無效"
echo ""

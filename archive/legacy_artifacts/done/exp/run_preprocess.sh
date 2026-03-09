#!/bin/bash

# Zero-Shot Speaker Denoising - 數據預處理腳本
# 目的: 一次性計算所有音頻的 tokens 和 speaker embeddings
# 優勢:
#   - 預處理一次，重複使用
#   - 訓練速度提升 8x
#   - GPU 利用率提升至 75-90%
#   - 節省 100 小時訓練時間

echo "========================================="
echo "Zero-Shot 數據預處理"
echo "========================================="
echo "此腳本將:"
echo "  1. 批量計算所有音頻的 WavTokenizer tokens"
echo "  2. 批量提取所有 speaker embeddings (ECAPA-TDNN)"
echo "  3. 保存到磁盤緩存 (./data/)"
echo ""
echo "配置:"
echo "  - Speakers: 18 人 (14 train, 4 val)"
echo "  - Sentences/speaker: 288"
echo "  - Materials: box, papercup, plastic, box2"
echo "  - Batch size: 32 (預處理批量大小)"
echo "  - 預計時間: 2-3 小時"
echo "  - 預計磁盤使用: 10-20 GB"
echo "========================================="
echo ""
echo "⚠️  注意："
echo "  - 確保有足夠的 GPU 記憶體"
echo "  - 確保有足夠的磁盤空間 (~20 GB)"
echo "  - 此步驟僅需執行一次"
echo "========================================="
echo ""

# 確認是否繼續
read -p "是否繼續? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "開始預處理..."
echo ""

# 設置 PYTHONPATH 以確保能找到所有模組
export PYTHONPATH="/home/sbplab/ruizi/c_code:${PYTHONPATH}"

python preprocess_zeroshot_cache.py \
    --input_dirs ../../data/raw/box ../../data/raw/papercup ../../data/raw/plastic ../../data/clean/box2 \
    --target_dir ../../data/clean/box2 \
    --output_dir ./data \
    --max_sentences_per_speaker 288 \
    --batch_size 32 \
    --speaker_encoder ecapa \
    --speaker_dim 256 \
    --device cuda:0

echo ""
echo "========================================="
echo "預處理完成！"
echo "========================================="
echo "生成的緩存文件:"
echo "  - ./data/train_cache.pt: 訓練集緩存"
echo "  - ./data/val_cache.pt: 驗證集緩存"
echo "  - ./data/cache_config.pt: 配置信息"
echo ""
echo "磁盤使用: $(du -sh ./data 2>/dev/null | cut -f1)"
echo ""
echo "下一步:"
echo "  運行訓練腳本: bash run_zeroshot_full_cached.sh"
echo ""
echo "預期加速效果:"
echo "  - 訓練速度: 8x 更快"
echo "  - GPU 利用率: 75-90%"
echo "  - 100 epochs: 15 小時 (vs 115 小時)"
echo "========================================="

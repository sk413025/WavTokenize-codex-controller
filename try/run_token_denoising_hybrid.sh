#!/bin/bash
#
# Token Denoising Transformer 訓練腳本 (混合損失版本)
#
# 結合三種損失:
# 1. CrossEntropy Loss (token 準確度)
# 2. Content Consistency Loss (相同內容應相似) - 動態權重
# 3. Embedding L2 Loss (embedding 空間接近)
#
# 實驗目的:
# - 解決純 CrossEntropy 的問題 (token 準確度高但音頻質量差)
# - 借鑑 ttt2.py 的內容一致性理念 (相同句子應有相似表示)
# - 在 embedding 空間約束模型學習更好的語義表示
#
# 數據改進:
# - 使用全部 288 句 (之前只用了 100 句)
# - max_sentences_per_speaker=None
#

# 設置環境
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/sbplab/ruizi/c_code:$PYTHONPATH
export ONLY_USE_BOX_MATERIAL=true  # ✅ 只使用 box 材質

# 數據路徑 (與 ttt2.py 一致)
# 使用相對路徑，從 try/ 目錄執行
INPUT_DIRS=(
    "../data/raw/box"
)

TARGET_DIR="../data/clean/box2"

# 輸出目錄 (使用相對路徑)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="../results/token_denoising_hybrid_loss_${TIMESTAMP}"

# 模型參數 (與 Frozen Codebook 一致)
D_MODEL=512
NHEAD=8
NUM_LAYERS=6
DIM_FEEDFORWARD=2048
DROPOUT=0.1

# 訓練參數
BATCH_SIZE=8
NUM_EPOCHS=600
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01

# 混合損失權重
CE_WEIGHT=1.0           # CrossEntropy 固定權重
CONTENT_WEIGHT=0.5      # Content Consistency 最大權重 (會動態衰減)
EMBED_WEIGHT=0.3        # Embedding L2 固定權重
WARMUP_EPOCHS=50        # 前 50 epochs 強調內容一致性

# WavTokenizer 配置 (與 run_token_denoising_frozen_codebook.sh 一致)
WAVTOKENIZER_CONFIG="../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOKENIZER_CHECKPOINT="../models/wavtokenizer_large_speech_320_24k.ckpt"

# 創建輸出目錄
mkdir -p "$OUTPUT_DIR"

# 記錄配置
echo "========================================" | tee "$OUTPUT_DIR/run_config.txt"
echo "Token Denoising Transformer 訓練 (混合損失)" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "========================================" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "開始時間: $(date)" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "數據配置:" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  輸入目錄: ${INPUT_DIRS[@]}" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  目標目錄: $TARGET_DIR" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  每位語者句子數: 全部 (288句)" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "模型配置:" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  d_model: $D_MODEL" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  nhead: $NHEAD" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  num_layers: $NUM_LAYERS" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  dim_feedforward: $DIM_FEEDFORWARD" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "訓練配置:" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  batch_size: $BATCH_SIZE" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  num_epochs: $NUM_EPOCHS" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  learning_rate: $LEARNING_RATE" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "混合損失權重:" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  CrossEntropy: $CE_WEIGHT (固定)" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  Content Consistency: $CONTENT_WEIGHT (最大值，動態衰減)" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  Embedding L2: $EMBED_WEIGHT (固定)" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "  Warmup Epochs: $WARMUP_EPOCHS" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "" | tee -a "$OUTPUT_DIR/run_config.txt"

# 開始訓練
echo "開始訓練..." | tee -a "$OUTPUT_DIR/run_config.txt"

python train_token_denoising_hybrid.py \
    --input_dirs "${INPUT_DIRS[@]}" \
    --target_dir "$TARGET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --num_layers $NUM_LAYERS \
    --dim_feedforward $DIM_FEEDFORWARD \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --ce_weight $CE_WEIGHT \
    --content_weight $CONTENT_WEIGHT \
    --embed_weight $EMBED_WEIGHT \
    --warmup_epochs $WARMUP_EPOCHS \
    --use_content_aware \
    --content_ratio 0.5 \
    --min_content_samples 3 \
    --wavtokenizer_config "$WAVTOKENIZER_CONFIG" \
    --wavtokenizer_checkpoint "$WAVTOKENIZER_CHECKPOINT"

# 記錄結束時間
echo "" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "結束時間: $(date)" | tee -a "$OUTPUT_DIR/run_config.txt"
echo "========================================" | tee -a "$OUTPUT_DIR/run_config.txt"

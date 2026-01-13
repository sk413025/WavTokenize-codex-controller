#!/bin/bash
# exp_test: 淺層 LoRA 容量瓶頸測試
#
# 核心問題: LoRA 容量是否是淺層學習不足的原因?
#
# 實驗設計:
#   - 只訓練 L0-L4 (5 層)，凍結 L5-L17
#   - Loss: L4 輸出 MSE
#   - 測試 rank = 256/512/1024
#
# 預期結果:
#   - 如果 rank↑ → loss↓↓ → 容量不足
#   - 如果 rank↑ → loss 無明顯改善 → 問題不在容量

set -e

# 設定環境
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 進入專案目錄
cd /home/sbplab/ruizi/WavTokenize-feature-analysis

echo "=========================================="
echo "exp_test: 淺層 LoRA 容量瓶頸測試"
echo "=========================================="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo ""
echo "測試組別:"
echo "  1. Rank 256  (baseline)"
echo "  2. Rank 512  (2x)"
echo "  3. Rank 1024 (4x)"
echo ""
echo "每組執行 150 epochs"
echo "=========================================="

# ============================================
# Rank 256 (baseline)
# ============================================
echo ""
echo "[1/3] Running Rank 256..."
echo "=========================================="

python exp_test/train.py \
    --exp_name shallow_r256 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lr 1e-4 \
    --num_epochs 150 \
    --batch_size 8 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_advance_epochs 30 \
    2>&1 | tee exp_test/runs/shallow_r256.log

echo ""
echo "[1/3] Rank 256 完成!"
echo ""

# ============================================
# Rank 512 (2x)
# ============================================
echo ""
echo "[2/3] Running Rank 512..."
echo "=========================================="

python exp_test/train.py \
    --exp_name shallow_r512 \
    --lora_rank 512 \
    --lora_alpha 1024 \
    --lr 1e-4 \
    --num_epochs 150 \
    --batch_size 8 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_advance_epochs 30 \
    2>&1 | tee exp_test/runs/shallow_r512.log

echo ""
echo "[2/3] Rank 512 完成!"
echo ""

# ============================================
# Rank 1024 (4x)
# ============================================
echo ""
echo "[3/3] Running Rank 1024..."
echo "=========================================="

python exp_test/train.py \
    --exp_name shallow_r1024 \
    --lora_rank 1024 \
    --lora_alpha 2048 \
    --lr 1e-4 \
    --num_epochs 150 \
    --batch_size 8 \
    --curriculum_mode curriculum \
    --initial_phase 0.3 \
    --phase_advance_epochs 30 \
    2>&1 | tee exp_test/runs/shallow_r1024.log

echo ""
echo "[3/3] Rank 1024 完成!"
echo ""

# ============================================
# 比較結果
# ============================================
echo "=========================================="
echo "所有實驗完成! 比較結果:"
echo "=========================================="

echo ""
echo "Rank 256:"
cat exp_test/runs/shallow_r256/summary.json 2>/dev/null || echo "  (尚無結果)"

echo ""
echo "Rank 512:"
cat exp_test/runs/shallow_r512/summary.json 2>/dev/null || echo "  (尚無結果)"

echo ""
echo "Rank 1024:"
cat exp_test/runs/shallow_r1024/summary.json 2>/dev/null || echo "  (尚無結果)"

echo ""
echo "=========================================="
echo "結論判斷:"
echo "  - 如果 loss 隨 rank 增加明顯降低 → 容量是瓶頸"
echo "  - 如果 loss 無明顯改善 → 問題在其他地方"
echo "=========================================="

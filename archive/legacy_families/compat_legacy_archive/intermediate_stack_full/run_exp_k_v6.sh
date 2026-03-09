#!/bin/bash
# ============================================================
# Exp K v6: TracIn Support + L4 Weight Adjustment
# ============================================================
#
# V6 改進重點:
#   1. Checkpoint 保存: 每 10 epochs (用於 TracIn 診斷)
#   2. L4 權重調整: 1.0 → 0.5 (解決 L4 過擬合問題)
#   3. 只保存 LoRA 參數 (節省空間)
#   4. 其他設定與 V5 相同
#
# V5 vs V6 差異:
#   - L4 weight: 1.0 → 0.5 (NEW)
#   - Checkpoint: 無 → 每 10 epochs (NEW)
#   - 其他參數: 相同
#
# encoder.model 結構:
#   model[3]: SConv1d (Downsample) - L3 (w=0.3)
#   model[4]: ResBlock - L4 (w=0.5, 降低自 1.0)
#   model[5]: ELU (無效!)
#   model[6]: SConv1d (Downsample) - L6 (w=0.5)
#
# 預計產出:
#   - best_model.pt (最佳模型)
#   - checkpoints/checkpoint_epoch{010,020,...,300}.pt (30 個)
#   - history.json (訓練歷史)
#   - audio_samples/ (音檔樣本)
#
# 執行:
#   bash families/compat_legacy/intermediate_stack/run_exp_k_v6.sh [GPU_ID]
# ============================================================

set -e

# GPU 設定 (預設 GPU 0)
GPU_ID=${1:-0}

# 設定環境
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 實驗名稱與時間戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="exp_k_v6_${TIMESTAMP}"

echo "============================================================"
echo "Exp K v6: TracIn Support + L4 Weight Adjustment"
echo "============================================================"
echo "實驗名稱: ${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo ""
echo "V6 改進 (vs V5):"
echo "  - L4 weight: 1.0 → 0.5 (解決過擬合)"
echo "  - Checkpoint: 每 10 epochs (用於 TracIn)"
echo "  - 只保存 LoRA 參數 (節省空間)"
echo ""
echo "監督層配置:"
echo "  - L3: 0.3 (Downsample)"
echo "  - L4: 0.5 (ResBlock, 降低自 1.0)"
echo "  - L6: 0.5 (Downsample)"
echo "============================================================"

python families/compat_legacy/intermediate_stack/train_v6.py \
    --exp_name "${EXP_NAME}" \
    --seed 42 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --intermediate_weight 0.5 \
    --intermediate_weight_min 0.25 \
    --warmdown_epochs 50 \
    --intermediate_L3_weight 0.3 \
    --intermediate_L4_weight 0.5 \
    --intermediate_L6_weight 0.5 \
    --num_epochs 300 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 10 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --curriculum_start 0.3 \
    --curriculum_end 0.85 \
    --curriculum_epochs 200 \
    --save_checkpoint_every 10 \
    --save_lora_only \
    --save_audio_interval 50 \
    --use_amp \
    2>&1 | tee families/compat_legacy/intermediate_stack/runs/${EXP_NAME}.log

echo "============================================================"
echo "Exp K v6 完成!"
echo "結果保存於: families/compat_legacy/intermediate_stack/runs/${EXP_NAME}"
echo "Checkpoints: families/compat_legacy/intermediate_stack/runs/${EXP_NAME}/checkpoints/"
echo "============================================================"

#!/bin/bash
# ============================================================
# Exp K v5: Dynamic Intermediate Layer Supervision with Warmdown
# ============================================================
#
# V5 改進重點 (解決 V4 困難樣本導致中間層損失持續上升問題):
#   1. 延長 Curriculum 過渡期: 200 epochs (原 100)
#   2. 限制困難樣本比例: curriculum_end=0.85 (排除最困難 15%)
#   3. 動態中間層權重衰減: 0.5 → 0.25 (warmdown 50 epochs)
#   4. 監督層配置不變: L3(0.3) + L4(1.0) + L6(0.5)
#
# encoder.model 結構:
#   model[3]: SConv1d (Downsample) - L3
#   model[4]: ResBlock (L4)
#   model[5]: ELU (無效!)
#   model[6]: SConv1d (Downsample) - L6
#
# V4 問題分析:
#   - Epoch 100 後困難樣本加入，中間層 loss 持續上升
#   - L4: 0.71 → 0.83 (+17%)
#   - L6: 0.79 → 0.85 (+8%)
#   - Val Loss 也從 1.80 → 1.90，出現過擬合
#
# V5 解決方案:
#   - 更慢的 curriculum 過渡 (200 epochs)
#   - 排除最困難 15% 樣本
#   - Warmdown: curriculum 完成後降低中間層權重
#
# 執行:
#   bash families/compat_legacy/intermediate_stack/run_exp_k_v5.sh
# ============================================================

set -e

# 設定環境
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"

# 啟動 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# 實驗名稱與時間戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="exp_k_v5_${TIMESTAMP}"

echo "============================================================"
echo "Exp K v5: Dynamic Intermediate Supervision with Warmdown"
echo "============================================================"
echo "實驗名稱: ${EXP_NAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "時間: $(date)"
echo ""
echo "V5 改進:"
echo "  - 延長 curriculum: 200 epochs (原 100)"
echo "  - 限制困難樣本: curriculum_end=0.85"
echo "  - 動態權重: 0.5 → 0.25 (warmdown 50 epochs)"
echo "  - 監督層: L3(0.3) + L4(1.0) + L6(0.5)"
echo "============================================================"

python families/compat_legacy/intermediate_stack/train_v5.py \
    --exp_name "${EXP_NAME}" \
    --seed 42 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --intermediate_weight 0.5 \
    --intermediate_weight_min 0.25 \
    --warmdown_epochs 50 \
    --intermediate_L3_weight 0.3 \
    --intermediate_L4_weight 1.0 \
    --intermediate_L6_weight 0.5 \
    --num_epochs 300 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 0.1 \
    --curriculum_start 0.3 \
    --curriculum_end 0.85 \
    --curriculum_epochs 200 \
    --save_audio_interval 50 \
    --use_amp \
    2>&1 | tee families/compat_legacy/intermediate_stack/exp_k_v5.log

echo "============================================================"
echo "Exp K v5 完成!"
echo "結果保存於: families/compat_legacy/intermediate_stack/runs/${EXP_NAME}"
echo "============================================================"

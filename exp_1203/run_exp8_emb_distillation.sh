#!/bin/bash
# exp_1203/run_exp8_emb_distillation.sh
#
# ================================================================
# 實驗目的: 使用 Encoder 原始輸出 (emb) 計算 Loss
# ================================================================
#
# 問題診斷 (VQ_LOSS_TOKEN_ACC_DIAGNOSIS.md):
#   原本的 feature_extractor 返回 quantized (量化後特徵)
#   而不是 encoder 原始輸出 (emb)
#   
#   這導致:
#   1. Loss 無法直接監督 encoder 輸出
#   2. encoder 輸出移動到錯誤的 Voronoi 區域
#   3. VQ Loss 上升 + Token Acc 下降
#
# 解決方案:
#   使用新的 EmbDistillationLoss
#   直接讓 student_emb → codebook[teacher_codes]
#   
#   Loss = MSE(student_emb, codebook[teacher_codes])
#   
#   這樣 encoder 輸出會被訓練成「等於」Teacher 選的 codebook embedding
#   從而保證 argmin 會選對 token
#
# 配置:
#   - distance_loss_mode: emb_distillation (新模式！)
#   - emb_to_codebook_weight: 1.0 (主要 Loss)
#   - ce_token_weight: 0.0 (可選的 CE 輔助 Loss)
#   - feature_loss_weight: 0.0 (量化後特徵對齊，不使用)
#
# 預期結果:
#   - VQ Loss 下降或穩定
#   - Token Accuracy 上升或維持高水平
#   - 修正 Token Acc 暴跌問題
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="emb_distillation"

echo "=========================================="
echo "Running exp_1203: ${EXP_NAME}"
echo "Strategy: Direct Encoder Output Supervision"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Problem solved:"
echo "  - Original: Loss uses quantized features (after VQ)"
echo "  - Fixed: Loss uses encoder raw output (emb)"
echo ""
echo "Key insight:"
echo "  - student_emb → codebook[teacher_codes]"
echo "  - This ensures argmin selects the correct token"
echo ""
echo "Config:"
echo "  - distance_loss_mode: emb_distillation"
echo "  - emb_to_codebook_weight: 1.0"
echo "  - ce_token_weight: 0.0"
echo "  - feature_loss_weight: 0.0"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1203

python train.py \
    --exp_name ${EXP_NAME} \
    --distance_loss_mode emb_distillation \
    --emb_to_codebook_weight 1.0 \
    --ce_token_weight 0.0 \
    --feature_loss_weight 0.0 \
    --soft_dist_loss_weight 0.0 \
    --vq_loss_weight 0.0 \
    --temperature 1.0 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --save_interval 10 \
    --log_interval 50 \
    --num_workers 4

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "Results: experiments/${EXP_NAME}/"
echo ""
echo "Expected outcomes:"
echo "  - VQ Loss: should decrease or stay stable"
echo "  - Token Accuracy: should stay HIGH (not collapse)"
echo "  - This validates the fix for encoder output supervision"
echo "=========================================="

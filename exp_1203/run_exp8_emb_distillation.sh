#!/bin/bash
# exp_1203/run_exp8_emb_distillation.sh
#
# ================================================================
# 實驗目的: EmbDistillation (連續域) + VQ Loss (離散域)
# ================================================================
#
# 問題診斷 (VQ_LOSS_TOKEN_ACC_DIAGNOSIS.md):
#   原本的 feature_extractor 返回 quantized (量化後特徵)
#   而不是 encoder 原始輸出 (emb)
#
#   這導致:
#   1. Feature Loss 無法直接監督 encoder 輸出（梯度無法回傳）
#   2. encoder 輸出移動到錯誤的 Voronoi 區域
#   3. VQ Loss 上升 + Token Acc 下降
#
# 解決方案:
#   1. EmbDistillationLoss (連續域對齊)
#      - 直接讓 student_emb → codebook[teacher_codes]
#      - Loss = MSE(student_emb, codebook[teacher_codes])
#      - encoder 輸出被訓練成「靠近」Teacher 選的 codebook embedding
#
#   2. VQ Loss (離散域穩定)
#      - Loss = MSE(quantize.detach(), x)
#      - 讓 encoder 輸出穩定在當前選擇的 codebook embedding 附近
#      - 防止 encoder 輸出在 Voronoi 邊界附近震盪
#
# 配置:
#   - distance_loss_mode: emb_distillation
#   - emb_to_codebook_weight: 1.0 (連續域：拉向 teacher 的 code)
#   - vq_loss_weight: 1.0 (離散域：穩定當前選擇)
#   - ce_token_weight: 0.0
#   - feature_loss_weight: 0.0 (有架構缺陷，不使用)
#
# 預期結果:
#   - EmbDistillation: 讓 encoder 輸出接近正確的 codebook embedding
#   - VQ Loss: 穩定離散選擇，減少震盪
#   - Token Accuracy 上升或維持高水平
#
# ===========================u=====================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=1

# 實驗名稱
EXP_NAME="emb_distillation"

echo "=========================================="
echo "Running exp_1203: ${EXP_NAME}"
echo "Strategy: EmbDistillation (連續域) + VQ Loss (離散域)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Problem solved:"
echo "  - Original: Feature Loss uses quantized (gradient cannot flow back)"
echo "  - Fixed: EmbDistillation uses encoder raw output (emb)"
echo ""
echo "Loss combination:"
echo "  - EmbDistillation: student_emb → codebook[teacher_codes] (連續域對齊)"
echo "  - VQ Loss: encoder output → current codebook embedding (離散域穩定)"
echo ""
echo "Config:"
echo "  - distance_loss_mode: emb_distillation"
echo "  - emb_to_codebook_weight: 1.0 (連續域)"
echo "  - vq_loss_weight: 1.0 (離散域)"
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

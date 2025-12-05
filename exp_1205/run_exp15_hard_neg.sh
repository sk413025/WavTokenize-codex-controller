#!/bin/bash
# ================================================================
# exp15: Hard Negative Mining + CE (方案 C)
# ================================================================
#
# 核心改進：
#   - 只在最近的 K 個 token 上計算 CE
#   - 強制模型學會區分「容易混淆」的 token
#   - 不浪費計算在距離很遠的 token 上
#
# 診斷結果顯示：
#   - Student embedding 靠近某些 token，但不是正確的 token
#   - 需要讓模型學會區分「附近」的 token
#
# 預期效果：
#   - 更高效的訓練（只關注 hard negatives）
#   - Token Accuracy 提升
#
# ================================================================

# 設定 GPU
export CUDA_VISIBLE_DEVICES=2

# 實驗名稱
EXP_NAME="exp15_hard_neg"

echo "=========================================="
echo "Running exp_1205: ${EXP_NAME}"
echo "Strategy: Hard Negative Mining + CE"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Key insight from diagnosis:"
echo "  - Student is close to SOME tokens, just not the CORRECT one"
echo "  - Hard Negative Mining: focus on distinguishing nearby tokens"
echo "  - K = 100 nearest tokens"
echo "=========================================="

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1205

python train.py \
    --exp_name ${EXP_NAME} \
    --loss_type hard_neg \
    --hard_neg_k 100 \
    --temperature 1.0 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --batch_size 20 \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --save_interval 10 \
    --log_interval 50

echo "=========================================="
echo "Experiment ${EXP_NAME} completed!"
echo "Results: experiments/${EXP_NAME}/"
echo ""
echo "Compare with exp_1204:"
echo "  - exp11 (MSE + CE, all 4096 tokens): Val Acc ~2%"
echo "  - exp15 (Hard Neg CE, top 100):      check experiments/${EXP_NAME}/"
echo ""
echo "If Val Acc improves: Hard negative mining is effective!"
echo "=========================================="

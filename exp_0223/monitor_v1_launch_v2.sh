#!/bin/bash
# ============================================================
# 監控腳本：等待 v1 Decoder LoRA 訓練完成後自動啟動 v2
# 
# v1 process PID: 2150799
# v1 run dir: exp_0223/runs/decoder_lora_epoch_20260223_010247/
# v2 script: exp_0223/train_decoder_lora_v2.py
# 
# 檢測方式（雙重判定）：
#   1. PID 2150799 不再存在
#   2. final_model.pt 已生成（確認是正常結束而非被 kill）
#
# 用法：
#   nohup bash exp_0223/monitor_v1_launch_v2.sh > exp_0223/monitor.log 2>&1 &
# ============================================================

set -e

# ===== 配置 =====
V1_PID=2150799
V1_RUN_DIR="exp_0223/runs/decoder_lora_epoch_20260223_010247"
V1_FINAL_MODEL="${V1_RUN_DIR}/final_model.pt"
V1_BEST_MODEL="${V1_RUN_DIR}/best_model.pt"

# v2 啟動參數
V2_SCRIPT="exp_0223/train_decoder_lora_v2.py"
V2_DEVICE="cuda:1"
V2_EPOCHS=150
V2_BATCH_SIZE=8
V2_GRAD_ACCUM=2
V2_LR="1e-4"
V2_MIN_LR="1e-6"
V2_WARMUP=5

# 監控間隔（秒）
CHECK_INTERVAL=60

# ===== 工作目錄 =====
cd /home/sbplab/ruizi/WavTokenize-feature-analysis

# ===== 激活 conda 環境 =====
eval "$(conda shell.bash hook)"
conda activate test

echo "========================================"
echo "  Decoder LoRA v1 → v2 自動監控腳本"
echo "========================================"
echo "啟動時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "監控 PID: ${V1_PID}"
echo "v1 run dir: ${V1_RUN_DIR}"
echo "檢查間隔: ${CHECK_INTERVAL}s"
echo "========================================"

# ===== 監控循環 =====
while true; do
    # 檢查 PID 是否還存在
    if kill -0 ${V1_PID} 2>/dev/null; then
        # v1 仍在運行
        # 抓取最新 epoch 進度
        LATEST_EPOCH=$(grep -oP 'Epoch \K\d+(?=/150)' "${V1_RUN_DIR}/train.log" 2>/dev/null | tail -1 || echo "?")
        echo "[$(date '+%H:%M:%S')] v1 仍在訓練中... (epoch ${LATEST_EPOCH}/150)"
        sleep ${CHECK_INTERVAL}
    else
        # PID 已消失
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] v1 PID ${V1_PID} 已結束！"
        
        # 檢查是否正常完成（有 final_model.pt）
        if [[ -f "${V1_FINAL_MODEL}" ]]; then
            echo "[INFO] ✅ v1 正常完成 — 發現 final_model.pt"
        else
            echo "[WARN] ⚠️  v1 可能異常結束 — 未發現 final_model.pt"
            echo "[WARN] 仍然使用 best_model.pt 啟動 v2..."
        fi
        
        # 確認 best_model.pt 存在
        if [[ ! -f "${V1_BEST_MODEL}" ]]; then
            echo "[ERROR] ❌ best_model.pt 不存在！無法啟動 v2。"
            exit 1
        fi
        
        echo ""
        echo "========================================"
        echo "  啟動 v2 Decoder LoRA (MR-STFT + Mel)"
        echo "========================================"
        echo "啟動時間: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Resume from: ${V1_BEST_MODEL}"
        echo "Device: ${V2_DEVICE}"
        echo "Epochs: ${V2_EPOCHS}"
        echo "========================================"
        
        # 等待 10 秒讓 GPU 記憶體釋放
        echo "[INFO] 等待 10 秒讓 GPU 記憶體釋放..."
        sleep 10
        
        # 啟動 v2
        python ${V2_SCRIPT} \
            --mode epoch \
            --epochs ${V2_EPOCHS} \
            --device ${V2_DEVICE} \
            --batch_size ${V2_BATCH_SIZE} \
            --grad_accum ${V2_GRAD_ACCUM} \
            --learning_rate ${V2_LR} \
            --min_lr ${V2_MIN_LR} \
            --warmup_epochs ${V2_WARMUP} \
            --weight_decay 0.01 \
            --decoder_lora_rank 32 \
            --decoder_lora_alpha 64 \
            --decoder_lora_dropout 0.1 \
            --lambda_wav 1.0 \
            --lambda_stft 1.0 \
            --lambda_mel 45.0 \
            --save_checkpoint_every 10 \
            --save_audio_interval 25 \
            --eval_max_batches 30 \
            --resume_from "${V1_BEST_MODEL}"
        
        V2_EXIT_CODE=$?
        echo ""
        echo "========================================"
        echo "  v2 訓練結束"
        echo "========================================"
        echo "結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Exit code: ${V2_EXIT_CODE}"
        echo "========================================"
        
        exit ${V2_EXIT_CODE}
    fi
done

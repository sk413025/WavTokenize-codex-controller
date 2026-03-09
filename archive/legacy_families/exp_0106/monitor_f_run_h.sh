#!/bin/bash
# 監控 Exp F，完成後自動運行 Exp H

EXP_F_LOG="/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0106/runs/exp_f_diff_lr/train.log"
EXP_H_SCRIPT="/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0106/run_exp_h.sh"

echo "=========================================="
echo "監控 Exp F 訓練進度..."
echo "Exp F Log: $EXP_F_LOG"
echo "完成後將自動運行: $EXP_H_SCRIPT"
echo "=========================================="

while true; do
    # 檢查 exp_f 進程是否還在運行
    if ! pgrep -f "exp_f_diff_lr" > /dev/null; then
        echo ""
        echo "[$(date)] Exp F 進程已結束"

        # 確認是否正常完成 (檢查是否有 Epoch 300)
        if grep -q "Epoch 300" "$EXP_F_LOG" 2>/dev/null; then
            echo "[$(date)] Exp F 已完成 300 epochs，訓練成功！"
        else
            # 檢查最後一個 epoch
            LAST_EPOCH=$(grep -oP "Epoch \d+" "$EXP_F_LOG" 2>/dev/null | tail -1)
            echo "[$(date)] 警告: Exp F 可能未完成全部訓練 (最後: $LAST_EPOCH)"
        fi

        echo ""
        echo "=========================================="
        echo "[$(date)] 開始運行 Exp H..."
        echo "=========================================="

        # 運行 Exp H
        bash "$EXP_H_SCRIPT"

        echo ""
        echo "=========================================="
        echo "[$(date)] Exp H 已啟動完成"
        echo "=========================================="

        break
    fi

    # 顯示當前進度
    CURRENT_EPOCH=$(grep -oP "Epoch \d+/300" "$EXP_F_LOG" 2>/dev/null | tail -1)
    echo -ne "\r[$(date '+%H:%M:%S')] Exp F 進度: $CURRENT_EPOCH - 等待中...    "

    # 每 60 秒檢查一次
    sleep 60
done

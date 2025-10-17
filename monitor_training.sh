#!/bin/bash

# 訓練監控腳本
# 用途：實時監控背景訓練的進度、GPU使用情況和損失變化

echo "=========================================="
echo "🔍 WavTokenizer 訓練監控工具"
echo "=========================================="
echo ""

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. 檢查訓練進程
echo -e "${BLUE}📊 訓練進程狀態:${NC}"
TRAIN_PROCESS=$(ps aux | grep "wavtokenizer_transformer_denoising.py" | grep -v grep)
if [ -z "$TRAIN_PROCESS" ]; then
    echo -e "${RED}❌ 訓練進程未運行${NC}"
else
    echo -e "${GREEN}✅ 訓練進程正在運行${NC}"
    echo "$TRAIN_PROCESS" | awk '{printf "   PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
fi
echo ""

# 2. 檢查GPU使用情況
echo -e "${BLUE}🎮 GPU 使用情況:${NC}"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader | while IFS=, read -r idx name temp gpu_util mem_util mem_used mem_total; do
    echo "   GPU $idx: $name"
    echo "   溫度: $temp | GPU使用率: $gpu_util | 記憶體使用率: $mem_util"
    echo "   記憶體: $mem_used / $mem_total"
done
echo ""

# 3. 查找最新日誌文件
LATEST_LOG=$(ls -t logs/wavtokenizer_transformer_training_*.log 2>/dev/null | head -1)
BACKGROUND_LOG=$(ls -t logs/background_training_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo -e "${BLUE}📝 最新訓練日誌: ${NC}$LATEST_LOG"
    
    # 4. 提取訓練進度
    echo -e "${BLUE}📈 訓練進度:${NC}"
    CURRENT_EPOCH=$(grep -oP "Epoch \K\d+(?=/)" "$LATEST_LOG" | tail -1)
    TOTAL_EPOCHS=$(grep -oP "Epoch \d+/\K\d+" "$LATEST_LOG" | tail -1)
    if [ -n "$CURRENT_EPOCH" ]; then
        echo -e "   當前: Epoch ${GREEN}$CURRENT_EPOCH${NC}/$TOTAL_EPOCHS"
        PROGRESS=$((CURRENT_EPOCH * 100 / TOTAL_EPOCHS))
        echo -e "   進度: ${GREEN}${PROGRESS}%${NC}"
    else
        echo -e "   ${YELLOW}⏳ 正在初始化...${NC}"
    fi
    echo ""
    
    # 5. 最近的損失值
    echo -e "${BLUE}📉 最近損失值 (最新5個epoch):${NC}"
    grep "Train Loss:" "$LATEST_LOG" | tail -5 | while read -r line; do
        EPOCH=$(echo "$line" | grep -oP "Epoch \K\d+")
        TRAIN_LOSS=$(echo "$line" | grep -oP "Train Loss: \K[\d.]+")
        VAL_LOSS=$(echo "$line" | grep -oP "Val Loss: \K[\d.]+")
        echo "   Epoch $EPOCH - Train: $TRAIN_LOSS, Val: $VAL_LOSS"
    done
    echo ""
    
    # 6. 檢查是否有錯誤
    echo -e "${BLUE}⚠️  錯誤檢查:${NC}"
    ERROR_COUNT=$(grep -i "error\|exception\|failed" "$LATEST_LOG" | wc -l)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "   ${RED}發現 $ERROR_COUNT 個錯誤/異常${NC}"
        echo "   最近的錯誤:"
        grep -i "error\|exception\|failed" "$LATEST_LOG" | tail -3 | sed 's/^/   /'
    else
        echo -e "   ${GREEN}✅ 無錯誤${NC}"
    fi
    echo ""
    
    # 7. 最新日誌輸出 (實時)
    echo -e "${BLUE}📜 最新日誌 (最後20行):${NC}"
    echo "----------------------------------------"
    tail -20 "$LATEST_LOG" | sed 's/^/   /'
    echo "----------------------------------------"
    
else
    echo -e "${YELLOW}⚠️  找不到訓練日誌文件${NC}"
fi

# 8. 檢查輸出模型
echo ""
echo -e "${BLUE}💾 保存的模型檢查點:${NC}"
LATEST_RESULT_DIR=$(ls -td results/wavtokenizer_tokenloss_* 2>/dev/null | head -1)
if [ -n "$LATEST_RESULT_DIR" ]; then
    echo "   輸出目錄: $LATEST_RESULT_DIR"
    MODEL_COUNT=$(ls "$LATEST_RESULT_DIR"/*.pth 2>/dev/null | wc -l)
    echo -e "   已保存模型數量: ${GREEN}$MODEL_COUNT${NC}"
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "   最新保存的模型:"
        ls -lth "$LATEST_RESULT_DIR"/*.pth | head -3 | awk '{printf "   - %s (%s)\n", $9, $5}'
    fi
else
    echo -e "   ${YELLOW}⚠️  尚未找到輸出目錄${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}監控完成！${NC}"
echo "提示: 使用 'watch -n 10 bash monitor_training.sh' 可每10秒自動更新"
echo "提示: 使用 'tail -f $LATEST_LOG' 可實時查看日誌"
echo "=========================================="

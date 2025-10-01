#!/bin/bash

# GPU修復腳本 - 將CrossEntropy實驗遷移到正常的GPU

echo "====================================================="
echo "GPU修復腳本 - CrossEntropy實驗GPU遷移"
echo "====================================================="

# 檢查當前實驗狀態
echo "1. 檢查當前實驗進程..."
PID=$(ps aux | grep wavtokenizer_transformer_denoising | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "   找到實驗進程 PID: $PID"
    echo "   實驗正在運行中，需要停止並重新啟動"
else
    echo "   未找到運行中的實驗進程"
    exit 1
fi

# 備份當前檢查點
echo "2. 備份當前檢查點..."
BACKUP_DIR="results/crossentropy_exp_202509300125_resumed_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r results/crossentropy_exp_202509300125_resumed/* "$BACKUP_DIR/" 2>/dev/null || echo "   警告：無法完全備份，可能檢查點正在寫入"

# 停止當前實驗
echo "3. 停止當前實驗..."
kill -TERM $PID
sleep 10

# 檢查是否完全停止
if ps -p $PID > /dev/null; then
    echo "   進程仍在運行，強制停止..."
    kill -KILL $PID
    sleep 5
fi

echo "4. 實驗已停止"

# 創建新的恢復腳本使用正常GPU
echo "5. 創建修復後的恢復腳本..."
cat > resume_crossentropy_experiment_fixed.sh << 'EOF'
#!/bin/bash

# CrossEntropy實驗 - 修復GPU配置後恢復
# 使用GPU 2 (正常的RTX 2080 Ti)

# 設置實驗參數
EXP_ID="202509300125_resumed_fixed"
BATCH_SIZE=8
NUM_WORKERS=4
OUTPUT_DIR="results/crossentropy_exp_${EXP_ID}"
LOG_FILE="logs/crossentropy_experiment_${EXP_ID}.log"

# 設置檢查點路徑 - 從之前的恢復實驗繼續
RESUME_CHECKPOINT="results/crossentropy_exp_202509300125_resumed/best_model.pth"

# 環境變量設置
export TTT_BATCH_SIZE="$BATCH_SIZE"
export TTT_NUM_WORKERS="$NUM_WORKERS" 
export TTT_EXPERIMENT_ID="${EXP_ID}"
export INPUT_SAMPLE_RATE=16000
export CONTENT_BATCHING=true

# GPU 配置 - 使用GPU 2 (正常的RTX 2080 Ti)
export CUDA_VISIBLE_DEVICES=2

# 創建必要目錄
mkdir -p logs
mkdir -p results
mkdir -p "$OUTPUT_DIR"

echo "====================================================="
echo "CrossEntropy實驗恢復 - GPU修復版本"
echo "====================================================="
echo "- 實驗ID: $EXP_ID"
echo "- 恢復檢查點: $RESUME_CHECKPOINT" 
echo "- 從最新狀態繼續"
echo "- GPU使用: GPU 2 (正常的RTX 2080 Ti)"
echo "- 批次大小: $BATCH_SIZE"
echo "- 輸出目錄: $OUTPUT_DIR"
echo "- 日誌文件: $LOG_FILE"
echo "====================================================="

# 檢查GPU可用性
echo "檢查GPU 2可用性..."
CUDA_VISIBLE_DEVICES=2 python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
else:
    print('CUDA不可用！停止執行')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "GPU 2不可用，停止執行"
    exit 1
fi

# 檢查檢查點
if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "錯誤：找不到檢查點文件 $RESUME_CHECKPOINT"
    exit 1
fi

echo "檢查點文件存在: $RESUME_CHECKPOINT"

# 激活conda環境
echo "激活conda環境..."
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

# 啟動實驗
echo "開始CrossEntropy實驗恢復..."
python wavtokenizer_transformer_denoising.py \
    --d_model 128 \
    --nhead 2 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --dim_feedforward 256 \
    --max_length 256 \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 25 \
    --val_speakers girl9 boy7 \
    --train_speakers boy1 boy3 boy4 boy5 boy6 girl2 girl3 girl4 girl6 girl7 \
    --max_sentences_per_speaker 100 \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" 2>&1 | tee -a "$LOG_FILE"

echo "實驗完成！"
EOF

chmod +x resume_crossentropy_experiment_fixed.sh

echo "6. GPU修復腳本創建完成"
echo ""
echo "後續步驟："
echo "1. 執行: nohup ./resume_crossentropy_experiment_fixed.sh > gpu_fix.log 2>&1 &"
echo "2. 監控: tail -f logs/crossentropy_experiment_202509300125_resumed_fixed.log"
echo "3. 備份位置: $BACKUP_DIR"
echo ""
echo "====================================================="
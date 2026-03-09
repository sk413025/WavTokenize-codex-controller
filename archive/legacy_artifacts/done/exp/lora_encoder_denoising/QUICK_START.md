# LoRA Encoder Denoising - 快速開始指南

## 🚀 立即開始訓練

### 選項 1: Baseline 訓練 (單 GPU)

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising

# 啟動訓練
./tmux_baseline.sh

# 連接到訓練 session
tmux attach -t lora_baseline_r16_lr5e5

# 斷開 session (訓練繼續執行)
# 在 tmux 內按: Ctrl+B 然後按 D
```

### 選項 2: Batch 1 平行訓練 (雙 GPU，推薦)

**同時訓練**: Baseline (rank=16) + Small LoRA (rank=8)

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising

# 啟動雙 GPU 平行訓練
./tmux_batch1_parallel.sh

# 連接到訓練 session
tmux attach -t lora_baseline_r16_lr5e5  # Baseline
tmux attach -t lora_lora_r8_lr5e5       # Small LoRA

# 查看所有 sessions
tmux ls
```

---

## 📊 監控訓練

### 即時查看訓練 log

```bash
# Baseline
tail -f experiments/baseline_r16_lr5e5.log

# Small LoRA (batch 1)
tail -f experiments/lora_r8_lr5e5.log
```

### Tensorboard 可視化

```bash
# 在另一個 terminal 啟動 Tensorboard
tensorboard --logdir experiments/ --port 6006

# 瀏覽器打開: http://localhost:6006
# 或 SSH tunnel: ssh -L 6006:localhost:6006 user@server
```

### GPU 監控

```bash
# 即時監控 GPU 使用
watch -n 2 nvidia-smi

# 查看特定 GPU
watch -n 2 'nvidia-smi -i 1,2'
```

---

## 🔧 Tmux 常用指令

### 基本操作

```bash
# 列出所有 sessions
tmux ls

# 連接到 session
tmux attach -t lora_baseline_r16_lr5e5

# 在 session 內斷開 (訓練繼續)
Ctrl+B, D  # 先按 Ctrl+B，放開後按 D

# 刪除 session (停止訓練)
tmux kill-session -t lora_baseline_r16_lr5e5

# 刪除所有 sessions
tmux kill-server
```

### Session 內操作

```bash
# 在 tmux session 內:
Ctrl+B, D           # 斷開 session
Ctrl+B, [           # 進入 scroll mode (查看歷史)
  ↑/↓ 滾動
  q 退出 scroll mode

Ctrl+C              # 中斷訓練 (不推薦，會留下 zombie session)
```

---

## ✅ 訓練前檢查清單

- [x] GPU 狀態正常 (nvidia-smi)
- [x] 數據載入測試通過 (7,776 + 1,440 samples)
- [x] 測試訓練驗證成功 (commit f2c8efc)
- [ ] Tmux 已安裝 (`tmux -V`)
- [ ] 選擇訓練腳本
  - [ ] 單 GPU: `./tmux_baseline.sh`
  - [ ] 雙 GPU: `./tmux_batch1_parallel.sh`
- [ ] Tensorboard 啟動 (可選)

---

## 📁 訓練輸出結構

```
experiments/
├── baseline_r16_lr5e5/          # Experiment 目錄
│   ├── checkpoints/
│   │   ├── best.pt              # 最佳 checkpoint
│   │   ├── latest.pt            # 最新 checkpoint
│   │   └── epoch_XXX_loss_Y.pt  # Top-K checkpoints
│   ├── logs/
│   │   └── events.out.tfevents.*  # Tensorboard logs
│   └── config.json              # 訓練配置
├── baseline_r16_lr5e5.log       # 訓練 log (文本)
└── ...
```

---

## 🛑 停止訓練

### 優雅停止 (推薦)

```bash
# 連接到 session
tmux attach -t lora_baseline_r16_lr5e5

# 按 Ctrl+C 停止訓練
# 然後 exit 退出 session
```

### 強制停止

```bash
# 直接刪除 session
tmux kill-session -t lora_baseline_r16_lr5e5

# 停止所有訓練
tmux kill-session -t lora_baseline_r16_lr5e5
tmux kill-session -t lora_lora_r8_lr5e5
```

---

## ⏱️ 預估時間

| Experiment | Epochs | Batch Size | 預估時間 |
|------------|--------|------------|----------|
| Baseline (rank=16) | 50 | 8 | ~5-6 小時 |
| Small LoRA (rank=8) | 50 | 8 | ~5-6 小時 |
| Large LoRA (rank=32) | 50 | 8 | ~6-7 小時 |

**Batch 1 平行訓練**: ~5-6 小時 (同時完成 2 個實驗)

---

## 🔍 故障排除

### Tmux session 已存在

```bash
# 錯誤: session 'lora_xxx' already exists
# 解決:
tmux kill-session -t lora_xxx
./tmux_baseline.sh  # 重新執行
```

### 訓練卡住 / 無輸出

```bash
# 連接到 session 查看
tmux attach -t lora_baseline_r16_lr5e5

# 檢查是否在 data loading (第一個 epoch 會較慢)
# 查看 GPU 使用
nvidia-smi
```

### GPU OOM

```bash
# 修改腳本，減小 batch_size
# tmux_baseline.sh: BATCH_SIZE=8 → BATCH_SIZE=4
```

---

## 📚 詳細文件

- [FULL_TRAINING_PLAN.md](./FULL_TRAINING_PLAN.md) - 完整訓練策略 (4 Phases)
- [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) - 訓練指南 (350+ 行)
- [TEST_RESULTS.md](./TEST_RESULTS.md) - 測試結果報告

---

**準備好了！執行 `./tmux_batch1_parallel.sh` 開始訓練！** 🚀

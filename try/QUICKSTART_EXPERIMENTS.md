# 🚀 實驗快速開始指南

## 📋 背景

發現 `run_simple_efficient.sh` 訓練效率遠低於 `debug_single_sample.py`:

| 腳本 | Epoch 100 | Epoch 200 | 收斂速度 |
|------|-----------|-----------|----------|
| debug_single_sample.py | ~100% | - | ⚡ 快速 |
| run_simple_efficient.sh | ~30% | ~20% | 🐌 很慢 |

**目標**: 找出根本原因

---

## ⚡ 5分鐘快速開始

### 1️⃣ 切換到實驗分支

```bash
cd /home/sbplab/ruizi/c_code
git checkout experiment/training-speed-debug
```

### 2️⃣ 查看實驗清單

```bash
ls -lh try/experiment_*.sh
```

你會看到：
```
experiment_1_no_dropout.sh       # 驗證 Dropout 影響
experiment_2_no_weight_decay.sh  # 驗證 Weight Decay 影響
experiment_3_mini_dataset.sh     # 驗證資料量影響 ⭐ 推薦優先
```

### 3️⃣ 執行實驗 3 (最推薦)

```bash
# 最快 (~1-2小時)，且最可能找到答案
bash try/experiment_3_mini_dataset.sh
```

**為什麼推薦實驗 3？**
- ⚡ 最快：只用 14 個音檔，訓練速度快
- 🎯 最關鍵：資料量差異 (14 vs 240+) 理論上影響最大
- 📊 最直接：如果 Epoch 100 達到 90%+，就證明資料量是主因

### 4️⃣ 監控進度

```bash
# 開啟另一個終端，監控日誌
tail -f logs/exp3_mini_dataset_*.log
```

### 5️⃣ 查看結果

實驗完成後：
```bash
# 查看 Epoch 100 的 Accuracy
grep "Epoch 100" logs/exp3_mini_dataset_*.log | grep "Train"

# 查看 Loss 曲線
ls -lh results/exp3_*/loss_curves_*.png
```

---

## 📊 如何解讀結果

### 情境 A: 資料量是主因 ✅

```bash
# 如果看到這樣的結果：
Epoch 100/200
  Train - Total Loss: 0.0123, CE: 0.0123, Acc: 95.67%
```

**結論**: 資料量差異是主因
- debug_single_sample.py 快是因為只需記憶 14 個模式
- run_simple_efficient.sh 慢是因為需要泛化到 240+ 個模式

**建議**:
- 接受這是正常現象（泛化需要更多時間）
- 或者減少資料集大小來加速初期實驗

---

### 情境 B: 資料量不是主因 ⚠️

```bash
# 如果看到這樣的結果：
Epoch 100/200
  Train - Total Loss: 2.3456, CE: 2.3456, Acc: 35.67%
```

**結論**: 其他因素更重要（Dropout 或 Weight Decay）

**下一步**: 執行實驗 1 和 2
```bash
bash try/experiment_1_no_dropout.sh      # 驗證 Dropout
bash try/experiment_2_no_weight_decay.sh # 驗證 Weight Decay
```

---

## 🔬 進階：執行所有實驗

### 選項 1: 手動逐個執行 (推薦)

```bash
# 實驗 3 (優先)
bash try/experiment_3_mini_dataset.sh

# 等待完成後，執行實驗 1
bash try/experiment_1_no_dropout.sh

# 等待完成後，執行實驗 2
bash try/experiment_2_no_weight_decay.sh
```

### 選項 2: 批次執行 (需要 8-10 小時)

```bash
# 會依序執行所有實驗，總耗時約 8-10 小時
bash try/run_all_experiments.sh
```

**注意**: 批次執行適合在背景執行或使用 tmux/screen

---

## 📈 結果比較表格

執行完實驗後，填寫此表格：

| 實驗 | Dropout | Weight Decay | 資料量 | Epoch 100 Acc | Epoch 200 Acc | 結論 |
|------|---------|--------------|--------|---------------|---------------|------|
| Baseline (原版) | 0.3 | 0.01 | 240+ | ~30% | ~20% | - |
| 實驗 1 (無 Dropout) | 0.0 | 0.01 | 240+ | ? | ? | 待填 |
| 實驗 2 (無 Weight Decay) | 0.3 | 0.0 | 240+ | ? | ? | 待填 |
| 實驗 3 (小資料集) | 0.0 | 0.01 | 14 | ? | ? | 待填 |
| debug_single_sample | 0.0 | 0.0 | 14 | ~100% | - | 參考 |

**快速填表命令**:
```bash
# 提取 Epoch 100 數據
echo "實驗 1:" && grep "Epoch 100" logs/exp1_*.log | grep "Acc"
echo "實驗 2:" && grep "Epoch 100" logs/exp2_*.log | grep "Acc"
echo "實驗 3:" && grep "Epoch 100" logs/exp3_*.log | grep "Acc"

# 提取 Epoch 200 數據
echo "實驗 1:" && grep "Epoch 200" logs/exp1_*.log | grep "Acc"
echo "實驗 2:" && grep "Epoch 200" logs/exp2_*.log | grep "Acc"
echo "實驗 3:" && grep "Epoch 200" logs/exp3_*.log | grep "Acc"
```

---

## 🛠️ 常見問題

### Q1: 實驗 3 創建小資料集時找不到音檔？

**檢查路徑**:
```bash
ls -lh data/raw/box/nor_boy1_box_LDV_001.wav
ls -lh data/clean/box2/nor_boy1_clean_001.wav
```

**如果路徑不對，修改腳本**:
```bash
# 編輯 try/experiment_3_mini_dataset.sh
# 找到第 63-65 行，修改路徑
```

### Q2: GPU 記憶體不足？

**檢查 GPU 狀態**:
```bash
nvidia-smi
```

**減少 Batch Size**:
```bash
# 編輯腳本，將 BATCH_SIZE=14 改為 8
BATCH_SIZE=8
```

### Q3: 想提前終止實驗？

```bash
# 找到 Python 進程
ps aux | grep train_token_denoising_hybrid.py

# 終止進程 (替換 PID)
kill -SIGINT <PID>
```

### Q4: 如何查看中間結果？

```bash
# 查看最新的日誌
tail -100 logs/exp3_mini_dataset_*.log

# 查看特定 epoch
grep "Epoch 50" logs/exp3_*.log
```

---

## 📞 獲取幫助

### 查看完整文檔

```bash
cat try/EXPERIMENTS_README.md
```

### 查看實驗配置

```bash
cat try/experiments_config.json
```

### 檢查腳本內容

```bash
# 查看實驗 3 的完整腳本
cat try/experiment_3_mini_dataset.sh
```

---

## ✅ 檢查清單

執行實驗前：
- [ ] 已切換到 `experiment/training-speed-debug` 分支
- [ ] 已激活 conda `test` 環境
- [ ] GPU 有足夠記憶體 (>8GB)
- [ ] 確認資料路徑存在

執行實驗後：
- [ ] 記錄 Epoch 100 的 Accuracy
- [ ] 記錄 Epoch 200 的 Accuracy
- [ ] 查看 Loss 曲線圖
- [ ] 填寫結果比較表格
- [ ] 更新實驗結論到 REPORT.md

---

## 🎯 下一步

1. **執行實驗 3**: `bash try/experiment_3_mini_dataset.sh`
2. **等待 1-2 小時**
3. **查看結果**: `grep "Epoch 100" logs/exp3_*.log`
4. **根據結果決定是否執行實驗 1 和 2**

---

最後更新: 2025-10-30

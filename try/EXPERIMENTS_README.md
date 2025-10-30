# 訓練速度差異驗證實驗

## 🎯 實驗目的

驗證為何 `run_simple_efficient.sh` (600 epochs → 60% accuracy) 比 `debug_single_sample.py` (100 epochs → ~100% accuracy) 慢得多。

---

## 📊 基準對比

### Baseline 1: debug_single_sample.py
- **資料集**: 14 個固定音檔 (LDV_001)
- **Dropout**: 0.0
- **Weight Decay**: 0.0
- **Batch Size**: 14 (= 樣本數)
- **Learning Rate**: 3e-4
- **結果**: Epoch 100 → ~100% accuracy ✅

### Baseline 2: run_simple_efficient.sh (原版)
- **資料集**: ~240+ 個音檔 (完整 box 材質)
- **Dropout**: 0.3
- **Weight Decay**: 0.01
- **Batch Size**: 14
- **Learning Rate**: 3e-4
- **結果**: Epoch 200 → ~20% accuracy, Epoch 600 → ~60% accuracy ⚠️

---

## 🔬 實驗設計

### 實驗 1: 驗證 Dropout 影響

**假設**: Dropout=0.3 導致資訊損失，累積效應使訓練困難

**變數控制**:
- ✅ 移除: Dropout = 0 (vs 原版 0.3)
- ⏸ 保持: Weight Decay = 0.01
- ⏸ 保持: 完整資料集 (~240+ 音檔)
- ⏸ 保持: 其他參數

**預期結果**:
- 如果 Dropout 是主因 → Epoch 200 應達到 40-50% accuracy
- 如果影響中等 → Epoch 200 達到 25-30% accuracy
- 如果影響小 → Epoch 200 仍只有 ~20% accuracy

**執行命令**:
```bash
bash try/experiment_1_no_dropout.sh
```

**日誌位置**:
```
logs/exp1_no_dropout_*.log
results/exp1_no_dropout_*/
```

---

### 實驗 2: 驗證 Weight Decay 影響

**假設**: Weight Decay=0.01 持續抑制權重成長，減緩學習

**變數控制**:
- ✅ 移除: Weight Decay = 0 (vs 原版 0.01)
- ⏸ 保持: Dropout = 0.3
- ⏸ 保持: 完整資料集 (~240+ 音檔)
- ⏸ 保持: 其他參數

**預期結果**:
- 如果 Weight Decay 是主因 → Epoch 200 應達到 35-45% accuracy
- 如果影響中等 → Epoch 200 達到 25-30% accuracy
- 如果影響小 → Epoch 200 仍只有 ~20% accuracy

**執行命令**:
```bash
bash try/experiment_2_no_weight_decay.sh
```

**日誌位置**:
```
logs/exp2_no_weight_decay_*.log
results/exp2_no_weight_decay_*/
```

---

### 實驗 3: 驗證資料量影響 (關鍵實驗)

**假設**: 資料量差異 (14 vs 240+) 是主要原因

**變數控制**:
- ✅ 修改: 只用 14 個音檔 (vs 原版 240+)
- ✅ 修改: Dropout = 0 (促進過擬合，與 debug_single_sample.py 一致)
- ⏸ 保持: Weight Decay = 0.01
- ⏸ 保持: 其他參數

**預期結果**:
- 如果資料量是主因 → Epoch 100 應達到 >90% accuracy (接近 debug_single_sample.py)
- 如果不是主因 → Epoch 100 仍然 <50% accuracy

**執行命令**:
```bash
bash try/experiment_3_mini_dataset.sh
```

**日誌位置**:
```
logs/exp3_mini_dataset_*.log
results/exp3_mini_dataset_*/
data/mini_dataset/  # 自動創建的小資料集
```

---

## 📈 結果分析方法

### 1. 快速查看 Accuracy

```bash
# 實驗 1
grep "Epoch 200" logs/exp1_no_dropout_*.log | grep "Train"

# 實驗 2
grep "Epoch 200" logs/exp2_no_weight_decay_*.log | grep "Train"

# 實驗 3
grep "Epoch 100" logs/exp3_mini_dataset_*.log | grep "Train"
grep "Epoch 200" logs/exp3_mini_dataset_*.log | grep "Train"
```

### 2. 繪製對比圖

查看各實驗的 loss curves:
```bash
ls -lh results/exp*/loss_curves_*.png
```

### 3. 比較表格

| 實驗 | Dropout | Weight Decay | 資料量 | Epoch 200 Acc | 結論 |
|------|---------|--------------|--------|---------------|------|
| Baseline (原版) | 0.3 | 0.01 | 240+ | ~20% | - |
| 實驗 1 | 0.0 | 0.01 | 240+ | ? | 待填 |
| 實驗 2 | 0.3 | 0.0 | 240+ | ? | 待填 |
| 實驗 3 | 0.0 | 0.01 | 14 | ? | 待填 |
| debug_single_sample | 0.0 | 0.0 | 14 | ~100% | 參考 |

---

## 🎯 判斷標準

根據實驗結果，我們可以判斷主要影響因素：

### 情境 A: Dropout 是主因
```
實驗 1 (無 Dropout) → Epoch 200 達到 40%+
實驗 2 (無 Weight Decay) → Epoch 200 仍 ~20%
實驗 3 (小資料集) → Epoch 100 達到 90%+
```
**結論**: Dropout 抑制資訊傳遞 + 資料量多 = 雙重減緩

---

### 情境 B: 資料量是主因
```
實驗 1 (無 Dropout) → Epoch 200 仍 ~25%
實驗 2 (無 Weight Decay) → Epoch 200 仍 ~25%
實驗 3 (小資料集) → Epoch 100 達到 90%+
```
**結論**: 樣本多樣性導致需要更多 epochs 才能泛化

---

### 情境 C: 多重因素組合
```
實驗 1 (無 Dropout) → Epoch 200 達到 30%
實驗 2 (無 Weight Decay) → Epoch 200 達到 28%
實驗 3 (小資料集) → Epoch 100 達到 90%+
```
**結論**: Dropout、Weight Decay、資料量皆有影響，需綜合調整

---

## 🚀 執行順序建議

建議按以下順序執行（從最可能的假設開始）：

### 第一優先: 實驗 3 (資料量)
```bash
bash try/experiment_3_mini_dataset.sh
```
**原因**: 最可能的主因，且執行速度最快（樣本少）

### 第二優先: 實驗 1 (Dropout)
```bash
bash try/experiment_1_no_dropout.sh
```
**原因**: 理論上有明顯的資訊損失效應

### 第三優先: 實驗 2 (Weight Decay)
```bash
bash try/experiment_2_no_weight_decay.sh
```
**原因**: 影響相對較小，但仍值得驗證

---

## 📝 實驗記錄模板

請在執行完每個實驗後填寫：

### 實驗 1 結果

**執行時間**: YYYY-MM-DD HH:MM
**GPU 使用**: GPU #?
**訓練耗時**: ? 小時

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 50    | ?          | ?         | ?        | ?       |
| 100   | ?          | ?         | ?        | ?       |
| 150   | ?          | ?         | ?        | ?       |
| 200   | ?          | ?         | ?        | ?       |

**觀察**:
- [ ] 收斂速度比原版快？
- [ ] Loss 曲線更穩定？
- [ ] Accuracy 提升明顯？

**結論**: (待填)

---

### 實驗 2 結果

(同上格式)

---

### 實驗 3 結果

(同上格式，特別關注 Epoch 100)

---

## 💡 後續實驗建議

根據初步結果，可能需要進行的進階實驗：

### 實驗 4: 組合優化
如果實驗 1-3 都顯示有影響，測試最佳組合：
```bash
# Dropout=0, Weight Decay=0, 完整資料集
# 預期: 在 200-300 epochs 達到 >50% accuracy
```

### 實驗 5: 階段性調整
```bash
# Epoch 1-100: Dropout=0.1, Weight Decay=0.001
# Epoch 101-200: Dropout=0.2, Weight Decay=0.005
# 預期: 兼顧收斂速度與泛化能力
```

---

## 📞 問題排查

### 如果實驗失敗

1. **檢查 GPU 記憶體**:
```bash
nvidia-smi
```

2. **檢查資料路徑**:
```bash
ls -lh data/raw/box/*.wav | wc -l
ls -lh data/clean/box2/*.wav | wc -l
```

3. **檢查日誌錯誤**:
```bash
tail -50 logs/exp*_*.log
```

---

## 🎓 理論分析

### Dropout 的累積效應

4 層 Transformer，每層 Dropout=0.3：
```
Layer 1: 保留 70% 資訊
Layer 2: 保留 70% × 70% = 49% 資訊
Layer 3: 保留 49% × 70% = 34% 資訊
Layer 4: 保留 34% × 70% = 24% 資訊
```
**結果**: 只有 ~24% 完整資訊到達輸出層

### 資料量的影響

- **14 個樣本**: 模型只需「記憶」14 個模式
- **240 個樣本**: 模型需「泛化」到 240 個不同模式
- **學習難度**: ~17 倍差異

---

## ✅ 檢查清單

實驗執行前：
- [ ] 確認在正確的分支 (`experiment/training-speed-debug`)
- [ ] 確認 conda 環境已激活 (`test`)
- [ ] 確認 GPU 有足夠記憶體 (>8GB)
- [ ] 確認資料路徑正確

實驗執行後：
- [ ] 保存日誌檔案
- [ ] 檢查 Loss 曲線圖
- [ ] 記錄關鍵 Accuracy 數據
- [ ] 更新結果到 REPORT.md
- [ ] Commit 實驗結果

---

最後更新: $(date "+%Y-%m-%d %H:%M:%S")

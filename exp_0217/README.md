# exp_0217: T453 Token-Aware Curriculum Weighting

**日期**: 2026-02-11
**基礎**: exp_0216 (Aug + LoRA-64)
**新增**: T453 token 濃度加權採樣

---

## 問題背景

Token 453 (T453) 是 WavTokenizer codebook 中最常見的 token：
- 在 val clean_tokens 中平均佔 **19.45%**
- 在 train clean_tokens 中平均佔 **13.10%**
- T453 對應語音中較為靜態/低能量的片段

若訓練初期大量看到高 T453 樣本，可能導致：
1. 模型偏向輸出 T453（降低 token diversity）
2. 學不到完整的語音結構

---

## T453 分佈統計

| 指標 | Train | Val |
|------|-------|-----|
| 平均 T453 比例 | 13.10% | 19.45% |
| T453 > 10% 樣本 | 57.7% | 75.7% |
| T453 > 30% 樣本 | 6.8% | 23.1% |
| T453 最高比例 | 49.1% | 50.5% |

---

## 解決方案：動態加權採樣

### 加權公式

```
w(sample, epoch) = 1.0 - (1 - min_weight) × t453_ratio × (1 - epoch_progress)

epoch_progress = epoch / ramp_epochs   (clipped at 1.0)
```

### 效果

| Epoch | 高T453樣本(>30%) 平均 weight | 低T453樣本 平均 weight | 比值 |
|-------|---------------------------|----------------------|------|
| 0     | 0.697                     | 0.910                | 0.77 |
| 50    | 0.798                     | 0.940                | 0.85 |
| 100   | 0.899                     | 0.970                | 0.93 |
| 150+  | 1.000                     | 1.000                | 1.00 |

### 設定（預設）

```yaml
t453_min_weight: 0.2   # 最低相對採樣權重
t453_ramp_epochs: 150  # 線性從 min_weight → 1.0 的 epoch 數
```

---

## 改進策略（相對 exp_0216）

| 改變 | exp_0216 | exp_0217 |
|------|----------|----------|
| 採樣策略 | SNR Curriculum | T453 Weighted Sampling |
| T453 處理 | 無 | 初期降權，漸進升至平等 |
| 資料增強 | 4 種 | 4 種（相同） |
| LoRA rank | 64 | 64（相同） |

---

## 檔案結構

```
exp_0217/
├── README.md                    # 本文件
├── data_t453_weighted.py        # T453WeightedSampler + create_t453_weighted_dataloaders
└── train_t453_weighted.py       # 訓練腳本（待實作）
```

---

## 使用方式

### 在訓練腳本中整合

```python
from exp_0217.data_t453_weighted import (
    create_t453_weighted_dataloaders,
    make_train_loader,
)

# 初始化（只做一次，預計算 T453 ratios）
train_dataset, val_loader, t453_sampler = create_t453_weighted_dataloaders(
    TRAIN_CACHE, VAL_CACHE,
    batch_size=8,
    total_epochs=300,
    t453_min_weight=0.2,
    t453_ramp_epochs=150,
)

# 訓練迴圈
for epoch in range(300):
    # 每個 epoch 更新採樣器（重新計算 weight）
    train_loader = make_train_loader(
        train_dataset, t453_sampler, epoch,
        batch_size=8, num_workers=2
    )

    for batch in train_loader:
        ...
```

---

## 預期效果

1. **Token Diversity 改善**：減少模型偏向 T453 的傾向
   - 預期 Entropy: 9.3+ (vs exp_0216 的 9.21)
   - 預期 Top-10 mass: <13% (vs exp_0216 的 13.98%)

2. **Overfitting 持續控制**：T453 加權不影響正則化效果

3. **收斂穩定**：ramp 設計讓模型逐漸適應高 T453 樣本

---

## 創建

**創建**: 2026-02-11
**狀態**: 🟡 設計完成，等待訓練腳本

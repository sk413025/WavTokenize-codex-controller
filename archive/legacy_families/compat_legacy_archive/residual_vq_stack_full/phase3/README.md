# exp_0128 Phase 3: Residual Vector Quantization (RVQ)

## 📋 總覽

Phase 3 使用 **Residual Vector Quantization (RVQ)** 架構來解決 token collapse 問題。

### 為什麼需要 RVQ？

**Phase 1 & 2 失敗原因**：
- Phase 1 (Sampling): 資料調整無效 ❌
- Phase 2 (Loss & Codebook): 表面修復無效 ❌
  - Exp 3 (Entropy Regularization): 所有實驗 entropy 下降
  - Exp 4 (Codebook Refresh): 所有實驗 top-10 mass 上升

**根本問題**: Encoder 輸出空間坍縮 → 只映射到 ~740/4096 codes

**RVQ 解法**: 架構層面強制多樣性 ✅
- 多層殘差量化
- 每層獨立操作
- 無法繞過的約束

## 📁 文件結構

```
exp_0128/phase3/
├── PLAN.md                          # 完整實驗計畫與 RVQ 原理
├── README.md                        # 本文件
└── residual_vq/
    ├── models_rvq.py                # RVQ 模型實作
    ├── test_rvq.py                  # RVQ 測試腳本 ✅
    ├── train_rvq_short_run.py       # RVQ 訓練腳本 ✅
    ├── run_exp5a.sh                 # 實驗 5a 啟動腳本 ✅
    ├── run_exp5b.sh                 # 實驗 5b 啟動腳本 ✅
    └── run_exp5c.sh                 # 實驗 5c 啟動腳本 ✅
```

## 🎯 實驗設計

### Baseline (exp_k v6 @ epoch 300)
- Entropy: 6.07
- Top-10 Mass: 19.7%
- Strict Accuracy: 0.91%
- Used Codes: ~740/4096 (18%)

### 成功判準（更嚴格）
- ✅ Val entropy > **6.5** (比 baseline 6.07 更高)
- ✅ Val top-10 mass < **15%** (比 baseline 19.7% 更低)
- ✅ Val strict acc >= **0.82%** (90% of baseline)

### 實驗配置

| 實驗 | 層數 | 每層 Codebook | 總表達能力 | 策略 | GPU |
|------|------|---------------|-----------|------|-----|
| **5a** | 2 | 2048 | 2048² = 4.2M | 溫和 (驗證概念) | 0 |
| **5b** | 4 | 1024 | 1024⁴ = 1.1T | 中等 (推薦) | 0 |
| **5c** | 8 | 512 | 512⁸ = 5.3e21 | 激進 (最大多樣性) | 1 |

## 🚀 快速開始

### 1. 測試 RVQ 模組

```bash
# 測試 RVQ forward pass, codebook usage, gradients
CUDA_VISIBLE_DEVICES=0 python exp_0128/phase3/residual_vq/test_rvq.py
```

預期輸出：
```
============================================================
Test 1: RVQ Forward Pass
============================================================
✅ Forward pass successful!

============================================================
Test 2: RVQ Codebook Usage
============================================================
✅ Codebook usage analysis complete!

============================================================
Test 3: RVQ Gradients
============================================================
✅ Gradients flowing correctly!

============================================================
✅ All tests passed!
============================================================
```

### 2. 運行實驗

#### Exp 5a (2 層 RVQ - 溫和測試)

```bash
bash exp_0128/phase3/residual_vq/run_exp5a.sh
```

#### Exp 5b (4 層 RVQ - 推薦配置)

```bash
bash exp_0128/phase3/residual_vq/run_exp5b.sh
```

#### Exp 5c (8 層 RVQ - 激進測試)

```bash
bash exp_0128/phase3/residual_vq/run_exp5c.sh
```

### 3. 監控實驗

```bash
# 查看日誌
tail -f exp_0128/phase3/residual_vq/run_exp5b_*.log

# 或使用 watch
watch -n 5 'tail -20 exp_0128/phase3/residual_vq/run_exp5b_*.log'
```

## 📊 預期結果

### 如果成功 ✅

**指標改善**：
- Entropy: 6.07 → **> 6.5** ⬆️
- Top-10 Mass: 19.7% → **< 15%** ⬇️
- Used Codes: 740 → **> 1500** ⬆️

**證明**：
- RVQ 有效防止 collapse
- 架構層面的改進有效
- 可以進行更長時間的訓練

### 如果失敗 ❌

**可能原因**：
1. 訓練不穩定 → 降低學習率 / 增加 gradient clipping
2. 層數不夠 → 嘗試更多層 (12/16 層)
3. Encoder 問題太深 → 考慮 Encoder 架構改進

**下一步**：
- 方案 2: Commitment Loss 調整
- 方案 3: Encoder 輸出正則化
- 方案 4: 訓練策略改變（見 PLAN.md）

## 📈 結果分析

實驗完成後，檢查以下文件：

```
run_exp5b_TIMESTAMP/
├── config.json              # 實驗配置
├── summary.json             # 最終結果總結
├── metrics_history.json     # 完整 metrics 歷史
├── loss_history.json        # Loss 歷史
└── training_curves.png      # 訓練曲線圖
```

### 關鍵指標

**查看 summary.json**：
```json
{
  "final_metrics": {
    "entropy": 6.78,           // 目標 > 6.5
    "top_10_mass": 0.12,       // 目標 < 0.15
    "strict_accuracy": 0.0085, // 目標 >= 0.0082
    "used_codes": 1856,        // 目標 > 1500
    "usage_pct": 90.4          // 使用率
  },
  "success": true              // 是否達標
}
```

**查看 training_curves.png**：
- 左上: Total Loss (應該下降)
- 左中: Loss Components (main, intermediate, RVQ)
- 左下: Entropy (應該上升，超過 6.5 線)
- 右上: Top-10 Mass (應該下降，低於 15% 線)
- 右中: Strict Accuracy (應該維持在 0.82% 以上)
- 右下: Used Codes (應該增加)

## 🔧 訓練參數

所有實驗使用一致的訓練配置（與 baseline 相同）：

```python
steps = 1000                    # 快速驗證
batch_size = 8                  # 與 baseline 一致
grad_accum = 2                  # 有效 batch size = 16
lr = 1e-4                       # 學習率
eval_interval = 200             # 每 200 步評估
seed = 42                       # 固定隨機種子

# LoRA 配置
lora_rank = 256
lora_alpha = 512
intermediate_supervision = [3, 6]  # L4, L8

# Loss 配置
feature_weight = 1.0
triplet_weight = 1.0
layer_weights = {3: 0.5, 6: 0.5}
```

唯一變化的參數：
- `n_rvq_layers`: RVQ 層數
- `rvq_codebook_size`: 每層 codebook 大小

## 📚 詳細文檔

- **RVQ 原理和架構**: 請閱讀 [PLAN.md](PLAN.md)
- **測試結果**: 請查看 `test_rvq.py` 輸出
- **訓練細節**: 請參考 `train_rvq_short_run.py` 源碼

## 🤝 與其他 Phase 對比

| Phase | 方法 | 層級 | 結果 |
|-------|------|------|------|
| Phase 1 | Sampling 調整 | Data-level | ❌ 全部失敗 |
| Phase 2 Exp 3 | Entropy Regularization | Loss-level | ❌ Entropy 下降 |
| Phase 2 Exp 4 | Codebook Refresh | Codebook-level | ❌ Top-10 上升 |
| **Phase 3 Exp 5** | **RVQ** | **Architecture-level** | **⏳ 測試中** |

## ⚠️ 注意事項

1. **記憶體使用**: RVQ 會增加記憶體使用（多層 codebook）
   - 2 層: ~2GB 額外記憶體
   - 4 層: ~4GB 額外記憶體
   - 8 層: ~8GB 額外記憶體

2. **訓練時間**: 與 baseline 相比略慢（多層量化）
   - 預期增加 10-20% 訓練時間

3. **Decoder 兼容性**: 當前實驗不使用 decoder
   - 我們只評估 codebook usage 和 entropy
   - 如需音頻重建，需要訓練新的 decoder

## 📞 問題排查

### 問題 1: CUDA Out of Memory

**解決方案**:
```bash
# 減少 batch size
--batch_size 4 --grad_accum 4  # 維持有效 batch size = 16

# 或減少 RVQ 層數
--n_rvq_layers 2  # 從 4 層降到 2 層
```

### 問題 2: 訓練不穩定（Loss 爆炸）

**解決方案**:
```bash
# 降低學習率
--lr 5e-5  # 從 1e-4 降到 5e-5

# 增加 gradient clipping（已在代碼中設置為 1.0）
```

### 問題 3: Metrics 不改善

**可能原因**:
1. 檢查 RVQ commitment loss 是否過大
2. 檢查每層 codebook usage（應該分布均勻）
3. 確認 gradient 正常流動

## 🎉 預期時間線

- ✅ **Day 1** (2026-02-03): RVQ 模組實作 + 測試
- ⏳ **Day 2**: 運行 Exp 5a 驗證概念
- ⏳ **Day 3**: 運行 Exp 5b, 5c 完整測試
- ⏳ **Day 4**: 結果分析 + 報告

---

**建立時間**: 2026-02-03
**狀態**: 🟢 準備就緒 - 可以開始實驗！

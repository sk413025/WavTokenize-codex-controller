# 本周工作總結 (2026-01-29)

## 快速概覽

**時間**: 2026-01-27 至 2026-01-29 (3 天)
**狀態**: ✅ Phase 2 準備完成，進度報告已含視覺化證據
**下一步**: 執行 Phase 2 實驗（Entropy Reg + Codebook Refresh）

---

## 本周 Git Commits

### Commit 1: d0f9ecb (2026-01-27)
```
feat(exp_0125): 5-checkpoint TracIn 分析完成

- 擴充從 2-checkpoint 到 5-checkpoint
- 產出 tracin_scores_5ckpt.csv (10,001 rows)
- 更穩健的 proponents/opponents profiles
- 6 個文件，+25,070 insertions
```

### Commit 2: 997d357 (2026-01-29)
```
docs(exp_0128): Phase 1 完整結果報告與失敗原因分析

- Phase 1 兩實驗均失敗（TracIn-Weighted & Noise-Balanced）
- 關鍵發現：訓練過程導致 collapse
- 提出 Phase 2 方案（4 個方向）
- 1 個文件，+524 insertions
```

### Commit 3: 8a685f9 (2026-01-29)
```
feat(exp_0128): Phase 2 架構層面修復方案準備完成

- 實驗 3: Entropy Regularization (λ ∈ {0.01, 0.05, 0.1})
- 實驗 4: Codebook Refresh (interval ∈ {50, 100})
- 完整實現代碼和執行腳本
- 21 個文件，+4,662 insertions
```

### Commit 4: 50bbcfc (2026-01-29)
```
docs: 為本周進度報告添加視覺化證據支持

- 新增「關鍵視覺化證據」專章
- 嵌入 11 個圖表引用（7 組證據）
- 創建圖表總覽附錄
- 1 個文件，+101 insertions
```

---

## 核心成果

### 1. TracIn 診斷完成 ✅

**技術提升**:
- 5-checkpoint 分析（epoch 010/050/150/250/300）
- L_train + L_anchor 雙版本交叉驗證
- 音質評估交叉檢查
- Counterfactual 驗證

**核心發現**:
| 指標 | Proponents | Opponents | 全體 |
|------|------------|-----------|------|
| SNR (dB) | **-2.24** | -0.68 | -1.88 |
| papercup | **57%** | 9% | 33% |
| box | 29% | **58%** | 33% |

**視覺證據**: TracIn Influence vs SNR 散點圖清楚顯示高 influence 集中在低 SNR 區域

### 2. Phase 1 驗證完成 ❌

**測試方案**:
1. TracIn-Weighted Soft Reweighting (α=0.5)
2. Noise-Balanced Sampling (box:papercup = 1:1)

**結果對比**:
| 實驗 | Entropy | Top-10 Mass | Strict Acc |
|------|---------|-------------|------------|
| Baseline | 6.07 | 19.7% | 0.91% |
| 實驗 1 | 5.63 ⬇ | 29.0% ⬆ | 0.60% ⬇ |
| 實驗 2 | 5.56 ⬇ | 28.3% ⬆ | 0.52% ⬇ |

**震撼發現**:
- 未訓練的 LoRA (Step 0) 優於訓練 300 epochs 的模型
- Entropy: 6.26 > 6.07 (+0.19)
- Top-10 Mass: 11.8% < 19.7% (-7.9%)

**視覺證據**: 兩實驗訓練曲線驚人相似，證明問題不在採樣策略

### 3. Phase 2 準備完成 ✅

**高優先級方案**:

#### 實驗 3: Entropy Regularization
```python
loss_entropy = -λ * entropy
total_loss = loss_inter + loss_main + loss_entropy
```
- 3 個參數: λ ∈ {0.01, 0.05, 0.1}
- 3 個執行腳本準備就緒

#### 實驗 4: Codebook Refresh
```python
if step % interval == 0:
    refresh_unused_codes(threshold)
```
- 2 個參數組: (interval=100, threshold=10), (interval=50, threshold=5)
- 2 個執行腳本準備就緒

### 4. 本周進度報告 ✅

**報告規模**:
- 676 行，6,500+ 字
- 11 個圖表引用（7 組視覺化證據）
- 6 大部分 + 3 個附錄
- 24 KB Markdown 文件

**視覺化證據**:
1. TracIn Influence vs SNR (證明診斷有效性)
2. 實驗 1 訓練曲線（TracIn-Weighted）
3. 實驗 2 訓練曲線（Noise-Balanced）
4. 補充圖表：2-ckpt 對照、L_anchor 驗證、音質分析

---

## 關鍵洞察

### 最重要發現

**Token collapse 是訓練動態問題，而非數據分佈問題**

**證據鏈**:
1. ✅ 未訓練的 LoRA 優於訓練後（entropy 6.26 vs 6.07）
2. ✅ 兩種完全不同的採樣策略產生相同失敗（差異 < 2%）
3. ✅ Collapse 主要發生在 step 0-200（critical period）
4. ✅ Training loss 正常下降，但 collapse metrics 持續惡化

**結論**: 需要架構或 loss 層面的根本改變，採樣調整無效

### 方法學貢獻

**TracIn 診斷流程**:
- 5-checkpoint 穩健分析
- 雙版本交叉驗證（L_train + L_anchor）
- 音質評估交叉檢查
- Counterfactual 驗證避免誤判

**實驗設計**:
- 配置完全一致（僅 sampler 不同）
- 詳細訓練動態記錄（每 200 steps）
- 多維度失敗原因剖析
- 視覺化證據支持

---

## 產出統計

### 代碼與數據

| 類別 | 數量 | 說明 |
|------|------|------|
| 新增文件 | 28 | 21 (Phase 2) + 7 (其他) |
| 代碼行數 | 4,662+ | Phase 2 實現 |
| CSV 數據 | 10,001 rows | TracIn 5-ckpt scores |
| JSON 數據 | 14,379 lines | TracIn indices |
| 視覺化圖表 | 11 個引用 | 7 組證據 |

### 文檔

| 文檔 | 行數 | 說明 |
|------|------|------|
| 本周進度報告 | 676 | 完整工作記錄 |
| Phase 2 README | 374 | 實驗設計文檔 |
| Phase 1 RESULTS | 524 | 失敗分析報告 |
| TracIn CONCLUSION | 295 | 診斷結果報告 |

### Git History

| Metric | 數量 |
|--------|------|
| Total Commits | 4 |
| Files Changed | 29 (unique) |
| Total Insertions | ~30,000+ |
| Total Deletions | ~70 |

---

## 時間投入

| 日期 | 工作內容 | 時長 |
|------|---------|------|
| 2026-01-27 | TracIn 5-checkpoint 分析 | 8-10 小時 |
| 2026-01-28 | Phase 1 實驗執行 + 分析 | 8-10 小時 |
| 2026-01-29 | Phase 2 準備 + 進度報告 | 6-8 小時 |
| **總計** | | **22-28 小時** |

### 計算資源

- **TracIn 分析**: 1 GPU, 6-8 小時
- **Phase 1 實驗**: 2 GPU 並行, 2.5 小時/實驗
- **存儲**: ~3 GB (checkpoints + metrics + plots)

---

## 下一步行動

### 立即執行 (本周內)

**Phase 2 實驗啟動**:

```bash
# GPU 0
bash exp_0128/phase2/entropy_regularization/run_exp3a_lambda_0.01.sh &
bash exp_0128/phase2/entropy_regularization/run_exp3c_lambda_0.1.sh &
bash exp_0128/phase2/codebook_refresh/run_exp4a_interval_100_thresh_10.sh

# GPU 1
bash exp_0128/phase2/entropy_regularization/run_exp3b_lambda_0.05.sh &
bash exp_0128/phase2/codebook_refresh/run_exp4b_interval_50_thresh_5.sh
```

**預期時間**: 2-3 小時（並行執行）

### 成功判準

與 baseline 比較，需同時滿足：
```python
success = (
    entropy > 6.07 AND
    top_10_mass < 19.7% AND
    strict_acc >= 0.82%  # 90% of baseline
)
```

### 後續計劃

**如果成功**:
1. Full training (300 epochs, 2-3 天)
2. 組合測試（Entropy Reg + Codebook Refresh）
3. 超參數優化

**如果失敗**:
1. 測試方案 C (Lower LR: 1e-4 → 5e-5)
2. 測試方案 D (Smaller LoRA: rank 256 → 128)
3. 考慮長期方案（重訓 VQ Codebook, Multi-task Learning）

---

## 文件索引

### 主要文檔

- [本周進度報告](本周進度報告_0129.md) - 完整工作記錄（含視覺化證據）
- [Phase 2 README](exp_0128/phase2/README.md) - Phase 2 實驗設計
- [Phase 1 RESULTS](exp_0128/RESULTS.md) - Phase 1 失敗分析
- [TracIn CONCLUSION](exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md) - TracIn 診斷報告

### 實驗代碼

**Phase 2 - Entropy Regularization**:
- [train_entropy_reg.py](exp_0128/phase2/entropy_regularization/train_entropy_reg.py)
- [run_exp3a_lambda_0.01.sh](exp_0128/phase2/entropy_regularization/run_exp3a_lambda_0.01.sh)
- [run_exp3b_lambda_0.05.sh](exp_0128/phase2/entropy_regularization/run_exp3b_lambda_0.05.sh)
- [run_exp3c_lambda_0.1.sh](exp_0128/phase2/entropy_regularization/run_exp3c_lambda_0.1.sh)

**Phase 2 - Codebook Refresh**:
- [train_codebook_refresh.py](exp_0128/phase2/codebook_refresh/train_codebook_refresh.py)
- [run_exp4a_interval_100_thresh_10.sh](exp_0128/phase2/codebook_refresh/run_exp4a_interval_100_thresh_10.sh)
- [run_exp4b_interval_50_thresh_5.sh](exp_0128/phase2/codebook_refresh/run_exp4b_interval_50_thresh_5.sh)

### 視覺化證據

**TracIn 診斷**:
- [profiles_5ckpt/plots/influence_vs_snr.png](exp_0125/tracin_token_collapse_589e6d/profiles_5ckpt/plots/influence_vs_snr.png)
- [profiles_5ckpt/plots/influence_vs_energy.png](exp_0125/tracin_token_collapse_589e6d/profiles_5ckpt/plots/influence_vs_energy.png)

**Phase 1 實驗**:
- [實驗 1 訓練曲線](exp_0128/soft_reweighting/run_exp1_20260129_023536/training_curves.png)
- [實驗 2 訓練曲線](exp_0128/noise_balanced_sampling/run_exp2_20260129_022108/training_curves.png)

---

## 總結

### 成就

✅ **完成 TracIn 穩健性提升** (5-checkpoint + 多重驗證)
✅ **識別 collapse 根本原因** (訓練動態問題)
✅ **Phase 1 驗證完成** (排除採樣調整方案)
✅ **Phase 2 準備就緒** (5 個實驗可立即執行)
✅ **完整文檔記錄** (含視覺化證據支持)

### 貢獻

**理論層面**:
- 確立 token collapse 為訓練動態問題
- 揭示未訓練狀態優於訓練後狀態
- 識別 critical period (step 0-200)

**方法學層面**:
- 建立 TracIn 多 checkpoint 分析流程
- 設計有效的 counterfactual 驗證
- 詳細的訓練動態分析方法
- 視覺化證據支持體系

### 下周目標

🎯 **短期（1-2 天）**: 完成 Phase 2 驗證，判定成功/失敗
🎯 **中期（3-5 天）**: 根據結果決定 full training 或中優先級方案
🎯 **長期（1-2 週）**: 如需要，考慮根本性架構改變

---

**生成日期**: 2026-01-29
**作者**: Ruizi (實驗) + Claude (分析與文檔)
**狀態**: 準備執行 Phase 2 ✅

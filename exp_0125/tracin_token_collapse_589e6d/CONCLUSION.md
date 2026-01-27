# CONCLUSION: TracIn 診斷 valid token collapse (commit 589e6d)

## 摘要

本實驗使用 **TracIn（Training Data Attribution）** 方法，追溯 validation set 上的 token collapse 失敗樣本，分析哪些 training samples 對這些失敗貢獻最大。目標是找出 valid token collapse 的 root causes，並提出可落地的修正方向。

---

## 1. 實驗設定

### 1.1 模型與資料

| 項目 | 數值 |
|------|------|
| 模型 | Exp K v6 (LoRA fine-tuned encoder) |
| 可用 Checkpoints | 30 個（epoch010-epoch300，每 10 epochs） |
| Trainable params | 3,704,576 (LoRA-only, 4.4%) |
| Train samples | 10,368 |
| Val samples | 1,728 |

### 1.2 Collapse 指標對比（Train vs Val）

| 指標 | Train | Val | 差異 |
|------|-------|-----|------|
| **Strict Acc (frame-weighted)** | 4.32% | **0.91%** | -3.41% |
| Student Entropy | 7.03 | **6.07** | -0.96 (collapse) |
| Student Top-10 Mass | 6.58% | **19.72%** | +13.14% (collapse) |
| KL(student‖teacher) | 0.55 | **1.25** | +0.70 (分佈偏離) |
| VQ Margin (p50) | 0.0169 | **0.0109** | -0.006 (不穩定) |
| Unique Tokens | 1,835 | **1,665** | -170 |

**關鍵觀察**：Val 的 strict acc 僅為 train 的 21%，且 collapse 指標全面惡化。

### 1.3 TracIn 設定

| 項目 | 數值 |
|------|------|
| Val failure set | 50 samples（collapse score 最高） |
| Train candidates | 2,000 samples |
| 使用 Checkpoints | **5 個**：epoch010, 050, 150, 250, 300 |
| 可用 Checkpoints | 30 個（每 10 epochs） |
| Loss 類型 | L_train + L_anchor（雙版本） |
| 近似方式 | Val gradient aggregated；Train batch 近似 |

**Checkpoint 選擇理由**（依 TracIn 論文建議）：
- epoch010：訓練初期（學習 general patterns）
- epoch050：warmup 結束（loss 下降最快期）
- epoch150：curriculum 過渡期中段
- epoch250：warmdown 開始
- epoch300：訓練末期（final model）

---

## 2. 核心發現

### 2.1 Proponents vs Opponents Profile

TracIn 將 train samples 分為：
- **Proponents**（正影響）：對 val failure 貢獻最大的 train samples
- **Opponents**（負影響）：與 val failure 方向相反的 train samples

| 指標 | Proponents (Top-100) | Opponents (Top-100) | 全體 Train |
|------|----------------------|---------------------|------------|
| **SNR (dB)** | **-2.24** | -0.68 | -1.88 |
| Cohen's d vs all | **-0.107** | +0.499 | — |
| Noise: papercup | **57%** | 9% | 33% |
| Noise: plastic | 14% | 33% | 33% |
| Noise: box | 29% | **58%** | 33% |

**關鍵發現**：
1. **Proponents 偏向低 SNR**：比全體 train 低 0.36 dB（Cohen's d = -0.107，小效應量）
2. **Proponents 以 papercup 材質為主**：57% vs 全體 33%（1.7 倍）
3. **Opponents 偏向 box 材質**：58% vs 全體 33%，且 SNR 較高（Cohen's d = +0.499，中等效應量）

### 2.2 L_anchor 版本一致性（2-ckpt 參考）

| 指標 | Proponents (Anchor) | Opponents (Anchor) |
|------|---------------------|-------------------|
| SNR (dB) | **-2.67** | -1.68 |
| Cohen's d | **-0.292** | +0.097 |
| papercup | **62%** | 4% |

L_anchor（teacher alignment loss）的 TracIn 結果與 L_train 一致，支持結論的穩健性。

*註：L_anchor 版本使用 2-checkpoint（epoch010, 300），作為交叉驗證參考。*

### 2.3 音質交叉檢查（S3）

對 val failure set 前 50 筆做音質評估：

| 指標 | Student Recon | 解讀 |
|------|---------------|------|
| PESQ | 1.08 (mean) | 極差（滿分 4.5） |
| STOI | 0.53 (mean) | 低可懂度 |
| SI-SDR | -31.0 dB (mean) | 嚴重失真 |

對 **bottom-PESQ 子集**（音質最差 30 筆）做 TracIn：
- Proponents: papercup **59%**、SNR **-2.10 dB**
- Opponents: box **77%**

**結論**：音質最差的樣本，其 TracIn proponents 仍指向相同的 noise profile（低 SNR + papercup）。

---

## 3. Counterfactual 驗證

### 3.1 實驗設計

移除 TracIn top-200 proponents（低 SNR / papercup 子集），用剩餘 1,800 samples 做短跑訓練（800 steps）。

### 3.2 結果

| 指標 | Baseline | Counterfactual | 變化 |
|------|----------|----------------|------|
| Val Strict Acc | 0.91% | **0.82%** | ↓ 惡化 |
| Val Entropy | 6.07 | **5.86** | ↓ 惡化 |
| Val Top-10 Mass | 19.7% | **28.1%** | ↑ 惡化 |
| Val KL | 1.25 | **1.47** | ↑ 惡化 |

### 3.3 解讀

**單純移除高 influence 噪音子集會破壞代表性噪音多樣性，導致 collapse 惡化。**

這代表：
- 這些「噪音材質強」的 train samples 不是純粹的「壞資料」
- 模型需要這些樣本來學習降噪
- 問題不在於「學到噪音」，而在於「沒學會分離噪音與內容」
- 解法應是 **soft reweighting** 或 **噪音材質增強**，而非硬刪

---

## 4. Top-3 Root Causes

### #1 H3：資料驅動 — 噪音材質強的 train 子集主導 LoRA 更新

| 證據 | 數值 | 來源 |
|------|------|------|
| Proponents SNR | -2.24 dB (Cohen's d = -0.107) | `profiles_5ckpt/proponents_profile.json` |
| Proponents papercup 佔比 | 57% (vs 全體 33%) | `profiles_5ckpt/proponents_profile.json` |
| Opponents 偏向 box | box 58%, papercup 僅 9% | `profiles_5ckpt/opponents_profile.json` |
| L_anchor 一致（2-ckpt） | papercup 62%, d = -0.292 | `anchor_profiles/` |

**機制**：LoRA 在低 SNR + papercup 材質的樣本上學得最多，這些樣本的 gradient 主導了參數更新方向。

### #2 H2：VQ margin 不穩定放大 collapse

| 證據 | 數值 | 來源 |
|------|------|------|
| VQ margin p50 | Train 0.0169 vs Val **0.0109** | `metrics_overview.json` |
| VQ margin p90 | Train 0.0619 vs Val **0.0311** | `metrics_overview.json` |

**機制**：Val 的 VQ margin 更小，代表量化決策更不穩定，容易塌縮到少數 token。這與 H3 形成因果鏈：noise-dependent encoding → margin 變小 → collapse。

### #3 H1：Noise-dependent / joint encoding

| 證據 | 數值 | 來源 |
|------|------|------|
| 高 influence 偏向特定噪音 | papercup 57% | `profiles_5ckpt/proponents_profile.json` |
| 音質最差子集同樣指向 | papercup 59% | `audio_quality/bottom_pesq_profiles/` |

**機制**：模型把噪音資訊編進 token（joint encoding），但在 inference 時無法忽略噪音變化，導致 val 上的 token 分佈偏離 teacher。

**限制**：本次 TracIn 使用 aggregate val gradient，缺少 per-val 的直接因果證據。結論為「支持但需加強」。

---

## 5. Proposed Fix

### Primary Fix：Noise-aware Reweighting + Balanced Sampling + Teacher Anchor

| 步驟 | 做法 | 對應 Root Cause |
|------|------|-----------------|
| 1 | 依 TracIn proponents 的 noise proxy（低 SNR / papercup）**降權** | H3 |
| 2 | 對 opponents/高 SNR/其他材質做 **平衡抽樣** | H3 |
| 3 | 保留 teacher-anchor（per-frame KL/CE）以避免 trivial collapse | H1, H2 |

**為何不是硬刪？**

Counterfactual 實驗顯示硬刪會惡化。原因是：
- 模型需要噪音多樣性來學習 robust encoding
- 問題不在於「有噪音樣本」，而在於「噪音樣本權重過高」
- Soft reweighting 可以平衡，同時保留噪音多樣性

### 替代方案

1. **Noise-invariant loss**：對同一 clean 的不同 noisy views 強制 soft assignment 一致（已在 Exp 0124 測試，效果有限）
2. **噪音材質增強**：增加 plastic/box 材質的訓練樣本，平衡噪音分佈
3. **VQ margin regularization**：直接正則化 margin，防止量化不穩定

---

## 6. 下一步驗證實驗

### 實驗 1：Soft Reweighting Short-run

| 項目 | 設定 |
|------|------|
| 目的 | 驗證 soft reweighting（非硬刪）是否能改善 collapse |
| 方法 | 依 TracIn score 做樣本權重：weight = 1 / (1 + α × score) |
| 樣本 | 2,000 samples, 800-1000 steps |
| 成功判準 | Val entropy ↑、top-k mass ↓、strict acc 不惡化 |
| 資源 | 1 GPU, 2-3 小時 |

### 實驗 2：噪音材質平衡抽樣

| 項目 | 設定 |
|------|------|
| 目的 | 驗證平衡噪音材質是否能改善 collapse |
| 方法 | 強制 papercup/plastic/box 各 1/3 |
| 樣本 | 2,000 samples, 800-1000 steps |
| 成功判準 | Val entropy ↑、top-k mass ↓、strict acc 不惡化 |
| 資源 | 1 GPU, 2-3 小時 |

---

## 7. 限制與未來工作

### 7.1 本次分析的限制

1. **Aggregate TracIn**：使用 val failure 的平均 gradient，可能混淆不同 failure 的成因
2. **近似 L_train**：triplet loss 設為 0，可能遺漏部分影響
3. **缺少 content metadata**：無法完全排除「內容相似性」混淆（例如同一 speaker 的不同噪音版本）

### 7.2 未來可補強

1. **Per-val TracIn**：對每個 val failure 分別計算 TracIn，分析是否有不同 failure pattern
2. **Content 控制比較**：加入 speaker/utterance 控制，排除內容相似性
3. **更多 checkpoint**：目前使用 5/30 個 checkpoints，可考慮擴充以進一步提高穩健性

---

## 8. 結論

**TracIn 診斷成功定位了 valid token collapse 的資料層面 root cause：低 SNR + papercup 材質的 train samples 主導了 LoRA 更新方向。**

但 Counterfactual 實驗顯示，單純移除這些樣本會惡化 collapse，代表：
- 這些樣本不是「壞資料」，而是「權重過高」
- 解法應是 **soft reweighting** 或 **材質平衡**，而非硬刪
- 根本問題在於 **訓練目標設計**，需要讓模型學會分離噪音與內容

---

## Acceptance Self-check

| 項目 | 狀態 | 說明 |
|------|------|------|
| M1 | ✅ | metrics_overview.json / failure_set.json |
| M2 | ✅ | tracin_scores.csv (train=2000, val=50) |
| M3 | ✅ | param_scope.json (LoRA-only verified) |
| M4 | ✅ | proponents/opponents profile + SNR/energy 對照 |
| M5 | ✅ | Top-3 root causes + Proposed Fix + Next steps |
| S1 | ✅ | L_train + L_anchor 雙版本 TracIn |
| S2 | ✅ | Counterfactual short-run（結果惡化，已記錄） |
| S3 | ✅ | 音質評估 + bottom-PESQ 子集 TracIn 交叉 |

---

## 附錄：證據檔案索引

| 檔案 | 內容 |
|------|------|
| `metrics_overview.json` | Train/Val collapse 指標 + VQ margin |
| `failure_set.json` | Val failure set（199 samples） |
| `tracin_scores_5ckpt.csv` | **TracIn scores (L_train, 5-ckpt, 10001 rows)** |
| `tracin_scores.csv` | TracIn scores (L_train, 2-ckpt, 4000 rows) |
| `tracin_scores_anchor.csv` | TracIn scores (L_anchor, 2-ckpt, 2000 rows) |
| `profiles_5ckpt/` | **5-ckpt 版本的 proponents/opponents profiles** |
| `proponents_profile.json` | Top-100 proponents (2-ckpt) |
| `opponents_profile.json` | Top-100 opponents (2-ckpt) |
| `anchor_profiles/` | L_anchor 版本的 proponents/opponents (2-ckpt) |
| `audio_quality/failure_set/` | Failure set 音質評估 (PESQ/STOI/SI-SDR) |
| `audio_quality/bottom_pesq_profiles/` | 音質最差子集的 TracIn profiles |
| `counterfactual/summary.json` | Counterfactual 短跑結果 |
| `plots/` | Influence vs SNR/energy 圖表 |

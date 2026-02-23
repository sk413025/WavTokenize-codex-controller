# Commit 5e859b0 實驗：Val 音質不足根因分析規劃（PLAN）

## 1) 背景與問題定義

- **目標 commit**: `5e859b0b8ae0bf54d7cf77449f239ae2aeaa0edb`  
  （`feat(exp_0216): 資料增強 + LoRA-64 實驗完成 (300 epochs) + 三實驗最終分析`）
- **主要輸出**: `exp_0216/runs/augmented_long_20260216/`
- **已知現象**：
  - overfitting 已顯著改善（best/final gap 很小）
  - 但 **val 音質仍未達預期**（best val MSE 約 `0.0381`，未突破目標 `<0.035`）
- **核心問題**：為什麼在「泛化穩定」後，驗證集主觀/客觀音質仍不夠好？

---

## 2) 分析目標

1. 釐清 val 音質瓶頸是否來自：
   - 資料分布差異（train/val）
   - 訓練目標與感知音質不一致（MSE vs PESQ/STOI）
   - 模型/量化器容量與解碼端限制（Single VQ + frozen decoder）
2. 建立可重複的診斷流程，產出可追蹤的證據與結論。
3. 輸出下一輪實驗的參數化決策（不是只給直覺建議）。

---

## 3) 根因假設樹（Hypothesis Tree）

### H1. 資料分布落差主導（Data Shift）
- val 中高 T453 比例樣本較多，造成 token/音質表現退化。
- val 的 SNR、時長、語音活動比例分布與 train 不一致。

### H2. 訓練目標與感知音質不一致（Objective Mismatch）
- `feature_mse` 改善不等於 PESQ/STOI 改善。
- 現行 loss 權重更偏向表徵對齊，未直接優化感知品質。

### H3. 架構上限（Capacity / Bottleneck）
- Single VQ（K=4096）在特定語音細節重建有上限。
- frozen decoder 對 student quantized 分布的適配能力不足。

### H4. 增強策略副作用（Augmentation Side Effect）
- augmentation 對 train 有效，但和 val 真實噪音型態仍有落差。
- 部分增強（如 crop/stretch）可能影響語音自然度泛化。

---

## 4) 分析分期與里程碑

### Phase A — 基線重建（0.5 天）
- 固化 commit 與 run metadata（config/summary/history/log）。
- 建立「epoch 1~300 核心指標時間線」。
- 產出：`baseline_metrics_table.md`

### Phase B — 音質客觀評估（1 天）
- 對 `epoch_050/100/150/200/222/250/300` 做 train/val PESQ、STOI。
- 對比 noisy→recon 的提升量（delta）而非只看絕對值。
- 產出：`audio_quality_by_epoch.json/.md`

### Phase C — 分層診斷（1 天）
- 依樣本屬性分桶：
  - T453 ratio（低/中/高）
  - SNR（低/中/高）
  - 音長（短/中/長）
- 評估各桶的 MSE/PESQ/STOI 差異與失敗案例集中區。
- 產出：`stratified_failure_report.md`

### Phase D — 假設驗證與決策（0.5 天）
- 對 H1~H4 做證據評分（支持/部分支持/不支持）。
- 輸出下一輪實驗設計（參數、預期改善幅度、風險）。
- 產出：`root_cause_decision.md`

---

## 5) 交付物（Deliverables）

1. `COMMIT_5e859b0_VAL_AUDIO_ANALYSIS_PLAN.md`（本文件）
2. `COMMIT_5e859b0_VAL_AUDIO_ANALYSIS_SPEC.md`
3. `COMMIT_5e859b0_VAL_AUDIO_ANALYSIS_ACCEPTANCE.md`
4. 分析產物目錄（建議）：`exp_0217/analysis_commit_5e859b0/`

---

## 6) 為什麼和研究目標直接相關

本研究核心是：**從 noisy speech 產生接近 clean quality 的離散表徵，且能泛化到驗證集**。  
`5e859b0` 已證明可以降低 overfitting，但若 val 音質仍不佳，代表「可泛化的音質」尚未達成。  
因此本分析不是補充工作，而是直接檢驗研究主張是否成立，並決定後續要優先調整：
- 資料分布策略（如 T453-aware sampling）
- 損失設計（加入感知目標）
- 量化/架構（Single VQ vs 更高容量）

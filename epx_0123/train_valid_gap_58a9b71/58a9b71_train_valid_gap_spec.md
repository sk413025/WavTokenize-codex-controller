# 分析規格：commit `58a9b71`（Exp K v5）Train/Valid accuracy 落差

## 1. 範圍（Scope）

本規格定義「如何以一致、可重現的方式」分析：
- 為何在 `commit 58a9b71` 導入的 Exp K v5 訓練流程中，出現 **train `masked_acc` 明顯高於 valid `masked_acc`** 的現象。

不在本規格內（但可作為後續）：直接修改訓練方法拿到更高 val acc 的完整研究迭代。

---

## 2. 主要輸入（Inputs）

### 2.1 版本與腳本
- commit：`58a9b71d2c9621b6485fbd019854b1526d9efea6`
- 訓練腳本：`exp_0112_intermediate/train_v5.py`
- 既有 run（基準樣本）：`exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848`
  - `best_model.pt`
  - `config.json`
  - `history.json`
  - `training_curves.png`

### 2.2 資料來源
以訓練腳本實際使用的 train/val cache 為準（通常由外部 config 提供，例如 `TRAIN_CACHE`、`VAL_CACHE`）。  
要求資料 loader 與訓練時一致（同一個 `Dataset`/`collate_fn`/resample/trim 策略），避免分析結論被 preprocessing 差異污染。

---

## 3. 指標定義（Metrics）

### 3.1 Strict masked token accuracy
- 定義：在有效 frame mask 下，`student_code == teacher_code` 的比例。
- 需同時回報兩種聚合方式（用來排除 H1）：
  - `acc_batch_mean`：每 batch 的 accuracy 取平均
  - `acc_frame_weighted`：`total_correct / total_frames`

### 3.2 Tolerant accuracy（用於對齊敏感度）
為了測試對齊偏移（H4），定義 tolerant 指標：
- `acc_tolerant_k`：允許 student token 序列在 ±k frames 內平移，取最佳對齊的 accuracy
- 建議至少回報：`k ∈ {1,2,3}`（stride=320 samples 約 13.3ms/frame）

### 3.3 Codebook 使用與多樣性
對 student/teacher 各自統計：
- `unique_codes`
- `entropy`（token histogram）
- `top_k_mass`（例如 top-10 token 佔比）
- `KL(student || teacher)`（或 Jensen–Shannon divergence）

### 3.4 Feature-space alignment（用於 H7）
在 final layer 及 v5 監督層（`[3,4,6]`）回報：
- `cos_sim(student_feat, teacher_feat)`（mask 後）
- `mse(student_feat, teacher_feat)`（可選）

---

## 4. 實驗設計（Protocol）

### 4.1 固定條件（Controls）
- checkpoint：至少跑 `best_model.pt`；若資源允許，加上 `final_model.pt`（或最後 epoch 的 checkpoint）
- 隨機種子：evaluation 階段固定 seed（資料載入可關閉 shuffle）
- model mode：evaluation 一律使用 `model.eval()` + `torch.no_grad()`

### 4.2 必要比較（Comparisons）
至少要產出以下 4 組結果（同一 checkpoint）：
1. Train split：strict（batch-mean + frame-weighted）
2. Val split：strict（batch-mean + frame-weighted）
3. Train split：tolerant（k=1,2,3）
4. Val split：tolerant（k=1,2,3）

加分（但建議做）：
- Train vs Val 的 token usage divergence（student/teacher 各自一份）
- Accuracy vs SNR / lag 的分桶曲線

---

## 5. 分析步驟（Steps）

### Step A：從 `history.json` 重建「gap 行為」
輸出：
- `gap_curve.png`：`train_masked_acc - val_masked_acc` 隨 epoch 變化
- `best_epoch`、`gap_at_best`、`gap_at_final`

### Step B：離線 eval（統一計算方式，排除 H1/H2）
輸出：
- `metrics_summary.json`：包含 strict/tolerant、batch-mean/frame-weighted、train/val
- `sanity_check.md`：紀錄「離線 eval」與訓練 log 的差異（若差異大，先追這個）

### Step C：資料難度與對齊分佈（對應 H3/H4）
輸出：
- `snr_hist_train_vs_val.png`
- `lag_hist_train_vs_val.png`
- `acc_vs_snr.png`、`acc_vs_lag.png`

說明：
- SNR 可用 noisy/clean 估計（`10*log10(Psignal/Pnoise)`）或使用 cache 內 metadata（若存在）。
- lag 可用 cross-correlation 抽樣估計；允許先抽樣（例如每 split 200 筆）再決定是否全量。

### Step D：token 多樣性/崩塌診斷（對應 H5）
輸出：
- `token_usage_train_vs_val.png`
- `token_usage_stats.json`（entropy、unique、top-k mass、divergence）

### Step E：feature-space 對照（對應 H7）
輸出：
- `feature_alignment_stats.json`（final layer + layers 3/4/6）
- 若 feature alignment 在 val 明顯不差但 token acc 很差，需在結論中明確標註「目標函數/指標不一致」。

---

## 6. 產出物格式（Outputs）

建議將所有結果集中於同一目錄（便於驗收與版本管理），例如：
- `exp_0112_intermediate/analysis/train_valid_gap_58a9b71/`
  - `metrics_summary.json`
  - `token_usage_stats.json`
  - `feature_alignment_stats.json`
  - `gap_curve.png`
  - `snr_hist_train_vs_val.png`
  - `lag_hist_train_vs_val.png`
  - `acc_vs_snr.png`
  - `acc_vs_lag.png`
  - `CONCLUSION.md`

`CONCLUSION.md` 必須包含：
- H1–H7 的逐項判定（支持/不支持/證據不足）
- 「主因排序」（Top-3）
- 對下一輪訓練/評估改動的最小建議（每項建議要能對應到某個假設）


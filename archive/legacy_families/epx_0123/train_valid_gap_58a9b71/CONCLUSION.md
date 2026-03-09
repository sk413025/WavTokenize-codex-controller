# CONCLUSION: commit 58a9b71 (Exp K v5) Train/Valid gap

## 核心結論（摘要）
- Gap 在訓練 log 與離線 eval 都存在，且量級一致：log best epoch 141 gap=0.01671；離線 eval（frame-weighted）gap=0.02469。
- **主要驅動因素是「token 分佈崩塌（mode collapse）」**，SNR 難度差與對齊偏移為次要因素。
- 報告主指標採 `acc_frame_weighted`；因 batch-mean 與 frame-weighted 幾乎一致，聚合方式不是主因。

> **2026-01-23 更新**：全量 SNR 分析（10,368/1,728 筆）顯示之前抽樣結果有偏差，SNR 差異較小（1.29 dB），主因排序已調整。
> **2026-01-23（Exp0123 後續驗證）**：global-shift tolerant 僅回補 val 約 +0.39%；SNR‑matched 仍保留 ~2.33% gap；val token entropy 與 acc 呈正相關（Pearson 0.269 / Spearman 0.217）。

## H1–H7 判定（證據 → 判斷）

### H1：聚合方式造成 train/val 不可比 → **不支持**
- 證據：`metrics_summary.json` 顯示 train 0.03396/0.03358、val 0.00899/0.00888（batch-mean / frame-weighted）；差異極小。
- 判斷：聚合方式不會導致 2% 以上 gap，非主因。

### H2：train 在 train()、val 在 eval() 造成落差 → **不支持**
- 證據：`sanity_check.md` 顯示離線 eval（eval mode）val ≈ log；gap 仍存在。
- 判斷：模式差異不是主因（但 train offline vs log 有差異，僅影響 train 絕對值）。

### H3：train/val 噪音難度分佈不同（SNR）→ **弱支持（僅次要）**
- 證據（抽樣 2000/500）：`snr_lag_stats.json` 顯示 SNR mean train 0.63 dB vs val -2.15 dB（差 2.78 dB）。
- **全量驗證（10,368/1,728）**：`full_snr_analysis/snr_lag_stats.json` 顯示 SNR mean train **-1.95 dB** vs val **-3.24 dB**（差 **1.29 dB**）。
- **SNR-matched 驗證（Exp0123 Step G）**：`snr_matched_stats.json` 顯示 train_matched strict fw **0.03222** vs val strict fw **0.00888**（gap **0.02334**），在分佈匹配下仍幾乎保留原 gap。
- 判斷：SNR 差異存在但無法解釋主要落差；僅能作為次要因素。

### H4：對齊偏移在 val 較嚴重 → **弱支持（次要）**
- 證據：`metrics_summary.json` tolerant k3 frame-weighted：val 0.00888 → 0.03210；`snr_lag_stats.json` 顯示 val lag mean 6.27 ms、std 41.59 ms（train mean -3.93 ms、std 26.21 ms）。
- 補充：此 tolerant 計算是「逐 frame 在 ±k shift 內取最大匹配」（per-frame max over offsets），屬於**偏樂觀的上界**，不等價於「整段序列做單一全域 shift 校正」。
- **全量驗證（500/500 samples）**：`full_snr_analysis/snr_lag_stats.json` 顯示 lag mean train **-1.9 ms** vs val **-0.1 ms**（幾乎相同），但 val std **46.5 ms** > train std **20.6 ms**。
- **Global-shift 驗證（Exp0123 Step F）**：`metrics_global_shift.json` 顯示 val strict fw **0.00888 → 0.01278**（+0.00390），train strict fw **0.03358 → 0.03440**；gap 僅縮小約 **0.00307**。
- 判斷：單一全域 shift 只能補回少量差距，對齊偏移非主因，但仍可能貢獻部分落差。

### H5：token 分佈/多樣性問題（mode collapse）→ **強支持** ⭐ 主因
- 證據：`token_usage_stats.json` 顯示 student entropy train 7.20 → val 5.89、top‑k mass 0.031 → 0.270、KL(student||teacher) 0.245 → 1.089。
- **歷史驗證**：exp_1226 的 `quick_token_acc_check.py` 診斷顯示 Student codes 集中在少數幾個值（top code 出現 24-34 次 vs Teacher 3-6 次），unique codes 也明顯少於 Teacher。
- **相關性驗證（Exp0123 Step H）**：`token_entropy_vs_acc_val.json` 顯示 entropy vs strict acc 相關（Pearson **0.269** / Spearman **0.217**），`token_usage_stats_val_full.json` 顯示 val entropy mean **6.67**、top‑k mass mean **0.304**。
- **小規模抑制實驗（Exp0123 Step I）**：
  - 低 λ：`ablation_anticollapse_summary_lowlambda.md`（λ=0.005/0.01，max_steps=200）未提升 val strict（0.008217/0.008684 vs 0.008693）。
  - 高 λ：`ablation_anticollapse_summary_highlambda.md`（λ=0.02/0.05/0.1，max_steps=400）val strict 分別 0.007987/0.006733/0.008225，亦未優於 baseline；entropy/top‑k mass 有些改善但不足以帶來 acc 提升。
  - **KL‑to‑teacher 正則**：`ablation_anticollapse_summary_kl.md`（λ=0.02/0.05/0.1，max_steps=400）val strict = 0.006954/0.007922/**0.009021**；KL(student||teacher) 明顯下降（1.189→0.784），顯示分佈對齊改善但 acc 提升有限。
  - 結論：KL 正則可改善分佈對齊，但對 strict acc 的提升仍小；若要更嚴謹驗證因果，需更長訓練或與其他正則/對齊策略組合。
- 判斷：低 entropy 對應低 acc，提供「collapse ↔ acc」的直接關聯證據；仍為 gap 主因。

### H6：資料切分/快取問題 → **不支持**
- 證據：`cache_overlap_report.md` 顯示 speaker_id、filename、noisy/clean path 皆無交集；noisy/clean path 組合亦為 0 交集。
- 補充：content_id 在 train/val 完全重疊，但 speaker_id 完全不重疊，代表可能是「相同內容、不同說話人」的設計，非樣本洩漏。

### H7：目標函數與指標不一致（feature alignment 良好但 token acc 差）→ **不支持**
- 證據：`feature_alignment_stats.json` 顯示 val 的 final layer feature alignment 變差（final cos 0.414 → 0.226；MSE 98.41 → 134.22）。屬於「feature 對齊也變差、token acc 也變差」的同向退化。
- 補充：中間層（3/4/6）在 cosine 指標下未必更差（本次抽樣甚至更高），但 MSE 對尺度非常敏感且未做 channel normalize，波動大；而 token codes 由 final encoder out 經 quantizer 得到，因此 H7 的判定主要以 **final layer 對齊是否良好** 為準。
- 判斷：不屬於「feature alignment 良好但 token acc 差」的指標不一致型態。

## Top-3 主因排序（含關鍵證據）— 2026-01-23（Exp0123 後續）

| 排名 | 假說 | 狀態 | 關鍵證據 |
|------|------|------|----------|
| **1** | **H5: Token 崩塌** | ⭐ 主因 | entropy-acc 相關（Pearson 0.269 / Spearman 0.217），val top‑k mass mean 0.304 |
| 2 | H4: 對齊偏移 | 次要 | global-shift k3 僅補回 val +0.00390（gap 仍 0.02162） |
| 3 | H3: SNR 難度差 | 次要 | SNR‑matched 後 gap 仍 0.02334 |

**排除的假說**：
- H1 聚合方式 ❌
- H2 train/eval 模式 ❌
- H6 資料洩漏 ❌
- H7 指標不一致 ❌

## 最小可行下一步（1–2 天內可驗證）
- **（優先）token 崩塌抑制**：加入 entropy regularizer 或 KL to teacher 的權重掃描，監控 top-k mass，短跑 1-2 個 epoch 驗證趨勢。
  - 參考 exp_1226 Exp65/Exp69 的 Anti-Collapse 實驗（λ=0.1 太強，λ=0.01 待驗證）。
- （次要）對齊敏感度：把 tolerant k∈{1,2,3} 加入常規評估作為輔助指標。
- （次要）SNR 分桶評估：在相同 SNR 範圍內比較 train/val acc，確認 gap 是否因難度差。

## 主要證據清單（可重現）
- `gap_curve.png`, `gap_summary.json`
- `metrics_summary.json`, `sanity_check.md`
- `snr_hist_train_vs_val.png`, `lag_hist_train_vs_val.png`, `snr_lag_stats.json`（抽樣）
- **`full_snr_analysis/snr_lag_stats.json`**（全量，2026-01-23 新增）
- `token_usage_train_vs_val.png`, `token_usage_stats.json`
- `metrics_global_shift.json`, `global_shift_hist_train_vs_val.png`
- `snr_matched_stats.json`, `snr_matched_eval.md`
- `token_usage_stats_val_full.json`, `token_entropy_vs_acc_val.json`, `token_entropy_vs_acc_val.png`
- `ablation_anticollapse_summary_lowlambda.md`, `ablation_anticollapse_summary_lowlambda.json`
- `ablation_anticollapse_summary_highlambda.md`, `ablation_anticollapse_summary_highlambda.json`
- `ablation_anticollapse_summary_kl.md`, `ablation_anticollapse_summary_kl.json`
- `feature_alignment_stats.json`
- `cache_overlap_report.md`, `cache_overlap_stats.json`

## 歷史實驗參考
- **exp_1226 Exp65**：Anti-Collapse (λ=0.1) → 0.69%，正則化太強
- **exp_1226 Exp69**：Anti-Collapse Light (λ=0.01) → 待執行
- **exp_1226 Frame-Tolerant**：只改善 +0.70%，時間偏移非主因

# Progress: Train/Valid Gap Analysis (commit 58a9b71)

## Checklist
- [x] Step A：從 `history.json` 重建 gap 行為（gap_curve.png + best_epoch/gap）
- [x] Step B：離線 eval（strict：batch-mean + frame-weighted；train/val；含 sanity_check.md）
- [x] Step C：資料難度/對齊分佈（SNR/lag + acc_vs_* 圖）
- [x] Step D：token 多樣性/崩塌（hist/entropy/unique/top-k mass/divergence）
- [x] Step E：feature-space 對照（final + layers 3/4/6）
- [x] 結論：完成 `CONCLUSION.md`（H1–H7 判定 + Top-3 + 下一步）

---

## Step A：gap 行為重建（完成）

結果摘要：
- `gap_curve.png` 已生成（train_masked_acc - val_masked_acc vs epoch）。
- best epoch = 141；train/val = 0.02570 / 0.00899；gap = 0.01671。
- final epoch = 300；train/val = 0.03046 / 0.00845；gap = 0.02200。

下一步：
- 進行離線 eval（Step B），統一 eval 模式與聚合方式，重現 gap 並產出 `metrics_summary.json` + `sanity_check.md`。

Blockers：
- 無。

Commands / Entrypoints：
- `python exp_0112_intermediate/analysis/train_valid_gap_58a9b71/stepA_gap_curve.py`

---

## Step B：離線 eval（完成：full）

結果摘要：
- 已產出 `metrics_summary.json`、`sanity_check.md`（checkpoint epoch=141；train/val 全量）。
- Strict acc（batch-mean / frame-weighted）：train 0.03396 / 0.03358；val 0.00899 / 0.00888；gap ≈ 0.0247。
- Tolerant acc 顯著提升 val（k3 frame-weighted=0.03210），顯示 strict 指標對對齊偏移敏感。
- 離線 eval 與訓練 log 對照：val 幾乎一致，gap 仍存在（見 `sanity_check.md`）。

下一步：
- 進行 Step C（資料難度/對齊分佈）。

Blockers：
- 無。

Commands / Entrypoints：
- `bash exp_0112_intermediate/analysis/train_valid_gap_58a9b71/run_stepB_full.sh`

---

## Step C：資料難度/對齊分佈（完成：抽樣）

結果摘要：
- 已產出 `snr_hist_train_vs_val.png`、`acc_vs_snr_train.png`、`acc_vs_snr_val.png`、`lag_hist_train_vs_val.png`、`acc_vs_lag_train.png`、`acc_vs_lag_val.png`、`snr_lag_stats.json`。
- SNR（train vs val, samples=2000/500）：mean 0.63 dB vs -2.15 dB，val 明顯更難。
- Lag（train vs val, samples=100/100）：mean -3.93 ms vs +6.27 ms；val 偏移量更大且變異更高（std 41.59 ms）。

下一步：
- 進行 Step D token 多樣性/崩塌分析。

Blockers：
- 無。

Commands / Entrypoints：
- `tmux new-session -d -s stepC_lag "bash -lc '... stepC_data_difficulty_alignment.py --snr_samples_train 2000 --snr_samples_val 500 --acc_snr_samples_train 2000 --acc_snr_samples_val 500 --lag_samples_train 100 --lag_samples_val 100 |& tee exp_0112_intermediate/analysis/train_valid_gap_58a9b71/stepC_lag_reduced.log'"`

---

## Step D：token 多樣性/崩塌（完成：抽樣）

結果摘要：
- 產出 `token_usage_train_vs_val.png`、`token_usage_stats.json`（train=2000 / val=500）。
- Student entropy (train/val): 7.20 → 5.89；top‑k mass: 0.031 → 0.270（val 明顯更集中）。
- KL(student||teacher): train 0.245 → val 1.089，val 分佈偏離 teacher 更大。

下一步：
- 進入 Step E（feature alignment）。

Blockers：
- 無。

Commands / Entrypoints：
- `tmux new-session -d -s stepD_token "bash -lc '... stepD_token_usage.py --max_train_samples 2000 --max_val_samples 500 |& tee exp_0112_intermediate/analysis/train_valid_gap_58a9b71/stepD_token_usage.log'"`

---

## Step E：feature-space 對照（完成：抽樣）

結果摘要：
- 產出 `feature_alignment_stats.json`（train=1000 / val=500）。
- Final layer cos(mean): train 0.414 → val 0.226（val 顯著較低）；MSE: 98.41 → 134.22。
- 補充：中間層（3/4/6）在 cosine 指標下未必更差（本次抽樣甚至更高），但 MSE 對尺度敏感且未做 channel normalize，波動大；本 step 的關鍵結論以 **final layer** 為主（因 token codes 由 quantizer(final out) 產生）。

下一步：
- 完成 Acceptance self-check。

Blockers：
- 無。

Commands / Entrypoints：
- `tmux new-session -d -s stepE_align "bash -lc '... stepE_feature_alignment.py --max_train_samples 1000 --max_val_samples 500 |& tee exp_0112_intermediate/analysis/train_valid_gap_58a9b71/stepE_feature_alignment.log'"`

---

## 結論：CONCLUSION.md（完成）

結果摘要：
- `CONCLUSION.md` 已完成，逐條判定 H1–H7，並提供 Top‑3 主因與最小下一步建議。
- 主因排序（更新後，見「全量 SNR 驗證」段落）：token 分佈崩塌（H5）> SNR 難度差（H3）> 對齊偏移敏感度（H4）。
- 重要證據來源：`metrics_summary.json`、`snr_lag_stats.json`、`token_usage_stats.json`、`feature_alignment_stats.json`。

下一步：
- 完成 Acceptance self-check（M1–M4）。

Blockers：
- 無。

Commands / Entrypoints：
- `cat exp_0112_intermediate/analysis/train_valid_gap_58a9b71/CONCLUSION.md`

---

## H6：cache/split 交集檢查（完成）

結果摘要：
- 產出 `cache_overlap_report.md`、`cache_overlap_stats.json`。
- speaker_id / filename / noisy_path / clean_path 皆無交集；noisy+clean path 組合交集為 0。
- content_id 完全重疊但 speaker_id 完全不重疊，較像「相同內容、不同說話人」設計而非樣本洩漏。

下一步：
- 無（已回填至 `CONCLUSION.md`）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && python exp_0112_intermediate/analysis/train_valid_gap_58a9b71/stepH6_cache_overlap.py`

---

## 全量 SNR 驗證（2026-01-23 新增）

結果摘要：
- 產出 `full_snr_analysis/snr_lag_stats.json`、`full_snr_analysis/snr_hist_train_vs_val.png` 等圖檔。
- **全量 SNR（10,368/1,728）**：train mean **-1.95 dB** vs val mean **-3.24 dB**（差 **1.29 dB**）。
- **之前抽樣有偏差**：抽樣顯示 train mean +0.63 dB，全量顯示 -1.95 dB（高估了 2.58 dB）。
- **全量 Lag（500/500）**：train mean -1.9 ms vs val mean -0.1 ms（幾乎相同），但 val std 46.5 ms > train std 20.6 ms。

結論：
- SNR 差異從主因降級為**次要因素**（1.29 dB 不足以解釋 2.47% gap）。
- 時間偏移差異不大，exp_1226 已驗證 Frame-Tolerant 只改善 +0.70%，降級為**次要因素**。
- **Token Collapse (H5) 升級為主因**。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && python exp_0112_intermediate/analysis/train_valid_gap_58a9b71/stepC_data_difficulty_alignment.py --acc_snr_samples_train 10368 --acc_snr_samples_val 1728 --lag_samples_train 500 --lag_samples_val 500 --num_workers 4 --output_dir exp_0112_intermediate/analysis/train_valid_gap_58a9b71/full_snr_analysis`

---

## Acceptance self-check（完成）
- M1: 已滿足（train/val 同時回報 batch-mean + frame-weighted；報告主指標採 frame-weighted，見 `CONCLUSION.md`）
- M2: 已滿足（`best_model.pt` 離線 eval 完成；gap 量級與方向明確，見 `metrics_summary.json`）
- M3: 已滿足（H1–H7 逐條判定，含支持/不支持/證據不足，見 `CONCLUSION.md`）
- M4: 已滿足（Top‑3 主因排序 + 最小下一步建議，見 `CONCLUSION.md`）
- **M5: 全量 SNR 驗證完成**（2026-01-23），主因排序已更新

---

### Follow-up (Exp 0123)
- [x] Step F: Global shift tolerant
- [x] Step G: SNR-matched eval
- [x] Step H: Token collapse robustness + correlation
- [x] Step I: Anti-collapse ablation (optional)

---

## Step F：Global shift tolerant（完成）

結果摘要：
- 產出 `metrics_global_shift.json`、`global_shift_hist_train_vs_val.png`。
- Strict → Global-shift(k3) frame-weighted：train 0.03358 → 0.03440（+0.00083）；val 0.00888 → 0.01278（+0.00390）。
- gap 由 0.02469 降至約 0.02162，僅回補約 0.39% 絕對值，說明單一全域 shift 只能解釋部分落差。

下一步：
- 進行 Step G（SNR-matched eval）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 python epx_0123/train_valid_gap_58a9b71/stepF_global_shift_tolerant_eval.py --batch_size 4 --num_workers 4 --progress_every 200 |& tee epx_0123/train_valid_gap_58a9b71/stepF_global_shift_tolerant_eval.log`

---

## Step G：SNR-matched eval（完成）

結果摘要：
- 產出 `snr_matched_stats.json`、`snr_matched_eval.md`（bin_width=2 dB，matched_train_len=1712）。
- train_matched strict frame-weighted = 0.03222；val strict frame-weighted = 0.00888；gap 仍約 0.02334。
- 即使 SNR 分佈匹配，gap 仍大幅存在，H3 無法解釋主要差距。

下一步：
- 進行 Step H（token collapse robust + correlation）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 python epx_0123/train_valid_gap_58a9b71/stepG_snr_matched_eval.py --batch_size 4 --num_workers 4 --progress_every 200 |& tee epx_0123/train_valid_gap_58a9b71/stepG_snr_matched_eval.log`

---

## Step H：Token collapse robustness + correlation（完成）

結果摘要：
- 產出 `token_usage_stats_val_full.json`、`token_entropy_vs_acc_val.json`、`token_entropy_vs_acc_val.png`（val full 1728）。
- val per-sample entropy vs strict acc 相關：Pearson 0.269、Spearman 0.217（正相關）。
- val entropy mean 6.67；top‑k mass mean 0.304，與低 acc 樣本呈集中趨勢。

下一步：
- 更新 `CONCLUSION.md`（強化 H5、下修 H3/H4）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 python epx_0123/train_valid_gap_58a9b71/stepH_token_entropy_vs_acc.py --batch_size 4 --num_workers 4 |& tee epx_0123/train_valid_gap_58a9b71/stepH_token_entropy_vs_acc.log`

---

## 結論更新（Exp0123）

結果摘要：
- `CONCLUSION.md` 已更新：H3/H4/H5 以 Step F/G/H 新證據強化；Top‑3 調整為 **H5 > H4 > H3**。
- H4 global-shift 僅補回 val +0.00390；H3 SNR‑matched 後 gap 仍 0.02334；H5 以 entropy‑acc 相關性強化主因結論。

下一步：
- （可選）Step I anti‑collapse 小規模 ablation。

Blockers：
- 無。

---

## Step I：Anti-collapse ablation（完成）

結果摘要：
- 產出 `ablation_anticollapse_summary.md`、`ablation_anticollapse_summary.json`（λ∈{0.0,0.005,0.01}；max_steps=200；train/val=2000/500）。
- val strict fw：0.008693（λ=0.0）→ 0.008217（λ=0.005）→ 0.008684（λ=0.01），短跑未見提升。
- val entropy：6.169（λ=0.0）→ 6.052（λ=0.005）→ 6.092（λ=0.01）；top‑k mass 與 KL 亦未改善為更分散。
- **高 λ 掃描（選項2）**：`ablation_anticollapse_summary_highlambda.md`（λ∈{0.02,0.05,0.1}；max_steps=400；train/val=2000/500）。val strict fw：0.007987 / 0.006733 / 0.008225（均未優於 baseline 0.008693）。entropy 與 top‑k mass 有輕微改善但 acc 未提升。
- **KL-to-teacher 正則（更嚴謹驗證）**：`ablation_anticollapse_summary_kl.md`（λ∈{0.02,0.05,0.1}；max_steps=400；train/val=2000/500）。val strict fw：0.006954 / 0.007922 / **0.009021**（λ=0.1 略高於 baseline 0.008693，但提升幅度小且仍低於 full‑val 0.00888）。KL(student||teacher) 明顯下降（1.189 → 0.784）。

下一步：
- 若要驗證 anti‑collapse 的真實效果，需更長訓練或不同正則設計（KL-to-teacher 已驗證，仍僅輕微改善）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python epx_0123/train_valid_gap_58a9b71/stepI_anticollapse_ablation.py --batch_size 2 --num_workers 0 --num_epochs 2 --max_steps 200 --max_train_samples 2000 --max_val_samples 500 --use_amp --gradient_accumulation_steps 2 |& tee epx_0123/train_valid_gap_58a9b71/stepI_anticollapse_ablation_retry3.log`
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python epx_0123/train_valid_gap_58a9b71/stepI_anticollapse_ablation.py --reg_type kl --entropy_weights 0.02,0.05,0.1 --batch_size 2 --num_workers 0 --num_epochs 2 --max_steps 400 --max_train_samples 2000 --max_val_samples 500 --use_amp --gradient_accumulation_steps 2 --output_root epx_0123/train_valid_gap_58a9b71/ablation_anticollapse_kl |& tee epx_0123/train_valid_gap_58a9b71/stepI_anticollapse_ablation_kl.log`

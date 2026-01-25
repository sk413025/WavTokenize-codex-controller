# Progress: Token Collapse Analysis (commit 27e564a)

## Checklist
- [x] Step A：Collapse overview（train/val；student/teacher；entropy/top‑k/unique/KL；strict acc）
- [x] Step B：Per-sample 證據（entropy/top‑k vs strict acc；case study top‑N）
- [x] Step C：條件化分析（至少兩個：speaker / SNR / energy / lag）
- [x] Step D：VQ margin 分析（d1/d2/margin；train vs val；與 acc/collapse 關聯）
- [x] Step E：Superposition 驗證（controlled pairs 或 probe）
- [x] Step F：CONCLUSION（逐條判定 + Top‑3 + Proposed Fix + 下一步）
- [x] Acceptance self-check（MUST/SHOULD/COULD）

---

## Step A：Collapse overview（完成）

結果摘要：
- 產出 `metrics_collapse_overview.json`（train/val 全量；seed=42；best_model.pt）。
- Train strict acc fw = 0.03358；Val strict acc fw = 0.00888（gap 明顯）。
- Val collapse 指標惡化：student entropy 5.91（train 6.91）、top‑k mass 0.251（train 0.096）、KL(student||teacher) 1.528（train 0.702）。

下一步：
- 進行 Step B（per-sample entropy/top‑k vs acc + case study）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0124/token_collapse_27e564a/stepA_collapse_overview.py --batch_size 4 --num_workers 4 --progress_every 200 |& tee exp_0124/token_collapse_27e564a/stepA_collapse_overview.log`

---

## Step B：Per-sample 證據（完成）

結果摘要：
- 產出 `token_entropy_vs_acc_val.json`、`token_entropy_vs_acc_val.png`、`case_studies.md`（val full 1728）。
- entropy vs strict acc 相關：Pearson 0.269、Spearman 0.217（正相關，但強度中等）。
- case studies 列出 top‑20 最崩樣本，顯示 collapse 不是單一極端樣本而是分佈性退化。

下一步：
- 進行 Step C（條件化分析：speaker / SNR / energy 等）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0124/token_collapse_27e564a/stepB_per_sample_analysis.py --batch_size 4 --num_workers 4 |& tee exp_0124/token_collapse_27e564a/stepB_per_sample_analysis.log`

---

## Step C：條件化分析（完成）

結果摘要：
- 產出 `collapse_by_speaker.json`、`collapse_by_snr.json`、`collapse_by_energy.json`（val full 1728）。
- speaker collapse 分佈偏平（各 speaker 的 collapse_score_mean 幾乎為 0），不支持單一 speaker shift 作為主因。
- SNR 與 energy bins 顯示 collapse/acc 有條件性差異，但非單一區間壟斷（需與 Step D/E 一起判定）。

下一步：
- 進行 Step D（VQ margin 分析）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0124/token_collapse_27e564a/stepC_conditioned_analysis.py --batch_size 4 --num_workers 4 |& tee exp_0124/token_collapse_27e564a/stepC_conditioned_analysis.log`

---

## Step D：VQ margin 分析（完成）

結果摘要：
- 產出 `vq_margin_stats_train_val.json`、`vq_margin_hist_train_val.png`（train/val 全量）。
- VQ margin（d2‑d1）在 val 顯著更小：mean 0.0138 vs train 0.0197；p50 0.0101 vs 0.0131。
- margin 變小代表量化不穩定可能加劇 strict acc 下降（H‑VQ 支持）。

下一步：
- 進行 Step E（Superposition 驗證）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0124/token_collapse_27e564a/stepD_vq_margin.py --batch_size 4 --num_workers 4 --progress_every 200 |& tee exp_0124/token_collapse_27e564a/stepD_vq_margin.log`

---

## Step E：Superposition 驗證（完成：controlled pairs）

結果摘要：
- 產出 `superposition_pair_tests.json`、`superposition_pair_plots.png`（N=30 clean × 3 noises × SNR{0,5,10}）。
- token_change_rate mean ≈ 0.897（std 0.029），同 clean 不同 noise 的 student tokens 變動極大。
- teacher_alignment_drop mean ≈ -0.002（std 0.0045），顯示對齊 drop 很小且波動（baseline 已低）。

下一步：
- 進行 Step F（CONCLUSION + Top‑3 + Proposed Fix + Acceptance self‑check）。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0124/token_collapse_27e564a/stepE_superposition_pairs.py --num_clean 30 --num_noise_per_clean 3 --snr_list 0,5,10 |& tee exp_0124/token_collapse_27e564a/stepE_superposition_pairs.log`

---

## Step F：CONCLUSION（完成）

結果摘要：
- `CONCLUSION.md` 已完成：Top‑3 root causes 排序、joint‑encoding 判定、noise‑invariant/disentanglement 判斷與 Proposed Fix。
- 明確提出 teacher‑anchored noise‑invariant training 作為 primary 解法，並列出 2 個可在 1–3 天內啟動的驗證實驗。
- 已完成 Acceptance self‑check（MUST/SHOULD/COULD）。

下一步：
- 若要落地驗證，開始執行 Proposed Fix 的短跑實驗。

Blockers：
- 無。

---

## Invariance short-run (Exp0124-2) — completed

結果摘要：
- 完成 λ=0.0/0.05/0.10；產出 `invariance_short_run/summary.{json,md}` 與各 lambda metrics。
- token_change_rate 僅小幅下降（0.9366 → 0.9218），未達 Go 門檻；collapse 指標有改善但 strict acc 僅小幅提升。
- **Global‑shift invariance (k=3)** 亦完成（`invariance_short_run_shift/summary.{json,md}`），token_change_rate 仍未顯著下降。
- `CONCLUSION.md` 已補上 Decision：No‑Go（需調整 invariance 設計或 pivot）。

下一步：
- 若要繼續，建議改用 feature‑level invariance 或轉向 probe/disentanglement。

Blockers：
- 無。

Commands / Entrypoints：
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0124/token_collapse_27e564a/invariance_short_run/run_invariance_short.py --lambdas 0.0 --max_steps 800 --max_train_samples 2000 --max_val_samples 500 --batch_size 2 --num_workers 2 --use_amp --gradient_accumulation_steps 2 |& tee exp_0124/token_collapse_27e564a/invariance_short_run/baseline_lambda0.log`

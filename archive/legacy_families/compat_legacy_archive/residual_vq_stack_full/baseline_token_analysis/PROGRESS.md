# Token Diversity: Epoch0 vs Epoch1 + Trend (Reproducible)

目標：用可重現的方式比較 **epoch 0 vs epoch 1** 的 token 多樣性與 code 使用率變化；若可取得多個 epoch 的點，產出趨勢圖。

參考單位定義：`exp_0128/baseline_token_analysis/DEFINITIONS.md`

---

## Run Context (fill by preflight)

- Timestamp (start): 2026-02-05T23:35:04-05:00
- `git rev-parse HEAD`: `9aa1331c8c474971583b4ef1e30a612d984437c2`
- Conda env: `test`
- Python: `3.10.13`
- Torch: `2.5.1` (`torch.version.cuda=11.8`)
- CUDA: `is_available=True`, `device_count=3`
- GPU (`nvidia-smi -L`):
  - GPU 0: NVIDIA GeForce GTX 1080 Ti
  - GPU 1: NVIDIA GeForce RTX 2080 Ti
  - GPU 2: NVIDIA GeForce RTX 2080 Ti
- Env:
  - `CUDA_VISIBLE_DEVICES=` (unset)
- Notes:
  - `import torch, numpy` OK（未觸發 OMP/SHM2 問題）

---

## Checkpoint Scan (baseline exp_k_v6)

- Path: `/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0112_intermediate/runs/exp_k_v6_20260125_234609_20260125_234613/checkpoints/`
- Findings:
  - Present: `checkpoint_epoch010.pt` ... `checkpoint_epoch300.pt` (saved every 10 epochs)
  - Missing: no `checkpoint_epoch0.pt` / `checkpoint_epoch1.pt` (or equivalent)
- Decision: **方案 B**（用 Phase3 step-based short-run 取得 epoch0/epoch1 對照）

---

## Metrics Definition (canonical)

對每個 epoch（或 eval step 對應的 epoch end）至少產出以下指標：

- `entropy_bits`
- `top_10_mass_pct`（注意：training `top10_mass` 常是 fraction → 需 ×100）
- `used_codes`
- `usage_pct`
- (optional) `top_1_mass_pct`, `top_50_mass_pct`, `top_100_mass_pct`

Splits（理想情況）：
- `train_student`, `train_teacher`, `val_student`, `val_teacher`

> 註：方案 B 的 Phase3/RVQ metrics 以 `layer0_*` 為主；teacher/student codebook space 不同時，僅做「各自分佈健康度」記錄，不做直接比較。

---

## Experiment / Analysis Tracker

| Item | Status | Command | output_dir | log | Start | End | Key result |
|---|---|---|---|---|---|---|---|
| Preflight | DONE | (see Timeline) |  |  | 2026-02-05T23:36:11-05:00 | 2026-02-05T23:36:11-05:00 | CUDA OK; 3 GPUs visible |
| SchemeB short (1 epoch) | DONE | `CUDA_VISIBLE_DEVICES=1 python exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 486 --eval_interval 486 ...` | `exp_0128/baseline_token_analysis/runs/schemeB_short_steps486_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_234533` | `.../train.log` | 2026-02-05T23:45:53-05:00 | 2026-02-05T23:50:49-05:00 | val: entropy 4.52→8.77, top10 70.0%→19.8%, used 63→1103 |
| SchemeB trend (N epochs) | DONE | `CUDA_VISIBLE_DEVICES=1 python exp_0128/phase3/residual_vq/train_rvq_short_run.py --steps 2430 --eval_interval 486 ...` | `exp_0128/baseline_token_analysis/runs/schemeB_trend_steps2430_eval486_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_235224` | `.../train.log` | 2026-02-05T23:52:41-05:00 | 2026-02-06T00:07:00-05:00 | val epoch5: entropy=8.42, top10=30.1%, used=995, mse=0.033 |
| Plot trends | DONE | `python exp_0128/baseline_token_analysis/analyze_rvq_by_epoch.py --out_trends_dir ...` | `exp_0128/baseline_token_analysis/trends/schemeB_trend_20260205_235224` |  | 2026-02-06T00:07:29-05:00 | 2026-02-06T00:07:30-05:00 | wrote 5 plots (val/train student) |

### Commands (exact)

#### SchemeB short (1 epoch)

```bash
source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test
CUDA_VISIBLE_DEVICES=1 python exp_0128/phase3/residual_vq/train_rvq_short_run.py \
  --steps 486 --batch_size 8 --grad_accum 2 --lr 1e-4 \
  --n_rvq_layers 4 --rvq_codebook_size 2048 \
  --rvq_update ema --ema_decay 0.99 --ema_eps 1e-5 --ema_dead_code_threshold 2 \
  --ema_usage_penalty 0.0 \
  --lambda_quant 1.0 --lambda_pre 0.0 --lambda_inter 0.5 --beta_commit 1.0 --lambda_codebook 1.0 \
  --inter_warmup_steps 0 \
  --eval_interval 486 --eval_max_batches 50 \
  --early_stop_on_collapse \
  --seed 42 --device cuda:0 \
  --output_dir exp_0128/baseline_token_analysis/runs/schemeB_short_steps486_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_234533
```

#### SchemeB trend (N epochs; N=5)

```bash
source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test
CUDA_VISIBLE_DEVICES=1 python exp_0128/phase3/residual_vq/train_rvq_short_run.py \
  --steps 2430 --batch_size 8 --grad_accum 2 --lr 1e-4 \
  --n_rvq_layers 4 --rvq_codebook_size 2048 \
  --rvq_update ema --ema_decay 0.99 --ema_eps 1e-5 --ema_dead_code_threshold 2 \
  --ema_usage_penalty 0.0 \
  --lambda_quant 1.0 --lambda_pre 0.0 --lambda_inter 0.5 --beta_commit 1.0 --lambda_codebook 1.0 \
  --inter_warmup_steps 0 \
  --eval_interval 486 --eval_max_batches 50 \
  --early_stop_on_collapse \
  --seed 42 --device cuda:0 \
  --output_dir exp_0128/baseline_token_analysis/runs/schemeB_trend_steps2430_eval486_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_235224
```

#### Extract epoch metrics + plots

```bash
source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test
python exp_0128/baseline_token_analysis/analyze_rvq_by_epoch.py \
  --run_dir exp_0128/baseline_token_analysis/runs/schemeB_trend_steps2430_eval486_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_235224 \
  --steps_per_epoch 486 \
  --out_trends_dir exp_0128/baseline_token_analysis/trends/schemeB_trend_20260205_235224
```

---

## Results

### Epoch0 → Epoch1 (Scheme B)

- Steps per epoch: `7776 / (batch=8 * grad_accum=2) = 486`
- Extract from `metrics_history.json` at `step=0` and `step=486`:
  - `layer0_entropy`
  - `layer0_top10_mass` (fraction) → `top_10_mass_pct = layer0_top10_mass * 100`
  - `layer0_used_codes`
  - `feature_mse`

#### Table (to be filled)

| epoch_end | step | layer0_entropy | top10_mass_pct | used_codes | usage_pct (used/2048) | feature_mse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 4.5176 | 70.0102 | 63 | 3.0762 | 0.0754 |
| 1 | 486 | 8.7722 | 19.8151 | 1103 | 53.8574 | 0.0385 |
| Δ (1-0) |  | +4.2546 | -50.1952 | +1040 | +50.7812 | -0.0369 |

#### Full splits (SchemeB trend run; epoch0/1)

Source: `.../schemeB_trend_steps2430_eval486.../delta_epoch0_epoch1.json`

| split | entropy_bits (0→1) | top10_mass_pct (0→1) | used_codes (0→1) | usage_pct (0→1) | feature_mse (0→1) |
|---|---:|---:|---:|---:|---:|
| val_student | 4.5176 → 8.5636 | 70.0102 → 25.9387 | 63 → 1014 | 3.0762 → 49.5117 | 0.0754 → 0.0368 |
| train_student | 4.6163 → 9.6739 | 63.0768 → 4.2697 | 63 → 1023 | 3.0762 → 49.9512 | 0.0978 → 0.0419 |
| val_teacher | 9.7008 → 9.7008 | 10.6388 → 10.6388 | 1779 → 1779 | 43.4326 → 43.4326 | N/A |
| train_teacher | 10.5487 → 10.5250 | 1.9653 → 2.1551 | 1826 → 1825 | 44.5801 → 44.5557 | N/A |

### Trend (Scheme B)

- Plots output: `exp_0128/baseline_token_analysis/trends/`

- Trend summary (val_student):
  - epoch0→1：快速去 collapse（entropy↑、top10↓、used_codes↑）
  - epoch2：最佳 top10（≈22.1%）/ used_codes≈1073
  - epoch3～5：val top10 有回升（≈30～35%）但未回到 collapse（used_codes 仍 ≈995～1089）
- Trend summary (train_student):
  - epoch1 之後 top10 維持在 ≈3～5%，used_codes ≈993～1095（更均勻於 val）

---

## Conclusion

1) **epoch0→epoch1（student / layer0）明顯更均勻**：entropy 大幅上升、top10 mass 大幅下降、used_codes/usage_pct 大幅上升；feature_mse 同步下降（更貼近 teacher feature）。

2) **Train/Val 方向一致但幅度不同**：
   - train_student 在 epoch1 之後非常均勻（top10≈3–5%）
   - val_student 在 epoch1 顯著改善，但後續 epoch3–5 有「top10 回升」現象（≈30–35%），仍未回到 collapse（used_codes ≈48–53%）

3) **未觀察到早期 collapse**（沒有出現 top10_mass≈100% 且 used_codes 很低的 epoch）。

**下一步（若目標是壓低 val 的 top10 回升）**
- 優先嘗試 `--ema_usage_penalty`（或 schedule）抑制 hot codes 漂移；並重跑多 seed 檢查趨勢是否一致。

---

## Timeline (append-only)

- 2026-02-05T23:35:04-05:00 init `PROGRESS.md`
- 2026-02-05T23:36:11-05:00 preflight: conda=test, torch=2.5.1+cu118, cuda_available=True, n_gpus=3, `git HEAD=9aa1331c8c474971583b4ef1e30a612d984437c2`
- 2026-02-05T23:45:53-05:00 SchemeB short start: `exp_0128/baseline_token_analysis/runs/schemeB_short_steps486_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_234533`
- 2026-02-05T23:50:49-05:00 SchemeB short end: wrote `metrics_history.json`, `summary.json`, `training_curves.png`, `final_model.pt`, `train.log`
- 2026-02-05T23:51:xx-05:00 SchemeB short analysis: wrote `epoch_metrics.{json,csv}` + `delta_epoch0_epoch1.json` via `exp_0128/baseline_token_analysis/analyze_rvq_by_epoch.py`
- 2026-02-05T23:52:41-05:00 SchemeB trend start: `exp_0128/baseline_token_analysis/runs/schemeB_trend_steps2430_eval486_L4_K2048_ema_th2_beta1p0_inter0p5_seed42_20260205_235224`
- 2026-02-06T00:07:00-05:00 SchemeB trend end: wrote `metrics_history.json`, `summary.json`, `training_curves.png`, `final_model.pt`, checkpoints
- 2026-02-06T00:07:29-05:00 SchemeB trend analysis+plots: wrote `epoch_metrics.{json,csv}` + `delta_epoch0_epoch1.json` + plots under `exp_0128/baseline_token_analysis/trends/schemeB_trend_20260205_235224`

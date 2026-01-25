# Progress: Invariance Short Run (Exp0124)

## Checklist
- [x] Step 0: Build fixed subsets (train=2000, val=500) and record indices/seed
- [x] Step 1: Training script ready (L_anchor + L_invar, lambda toggle)
- [x] Step 2: Run ablation (>= 3 lambdas)
- [x] Step 3: Summarize summary.{json,md}
- [x] Step 4: Update exp_0124/token_collapse_27e564a/CONCLUSION.md (Decision: Go/No-Go)

---

## Step 0: Fixed subsets (done)

Summary:
- Wrote subset indices to `invariance_short_run/runs/train_indices.json` and `invariance_short_run/runs/val_indices.json`.
- Seeds: train_seed=42, val_seed=43 (see run command).

Next:
- Continue baseline run (lambda=0).

Blockers:
- None.

Commands / Entrypoints:
- `python exp_0124/token_collapse_27e564a/invariance_short_run/run_invariance_short.py --lambdas 0.0 --max_steps 800 --max_train_samples 2000 --max_val_samples 500 --batch_size 2 --num_workers 2 --use_amp --gradient_accumulation_steps 2`

---

## Step 1: Training script (done)

Summary:
- Implemented `run_invariance_short.py` with L_anchor (base + intermediate) and L_invar (sym KL on soft assignments).
- Eval includes strict acc, collapse stats, token_change_rate, VQ margin.

Next:
- Run ablation lambdas after baseline completes.

Blockers:
- None.

Commands / Entrypoints:
- `exp_0124/token_collapse_27e564a/invariance_short_run/run_invariance_short.py`

---

## Step 2: Ablation runs (done)

Summary:
- λ=0.05/0.10 皆完成並產出 `runs/lambda_0.050/metrics.json`、`runs/lambda_0.100/metrics.json`。
- λ=0.05：val strict fw=0.006768；entropy=5.711；top‑k mass=0.286；KL=3.075；token_change_rate=0.9262；p50=0.008791。
- λ=0.10：val strict fw=0.006308；entropy=5.619；top‑k mass=0.312；KL=2.158；token_change_rate=0.9218；p50=0.008826。

Next:
- 更新 summary 與 Decision。

Blockers:
- None.

Commands / Entrypoints:
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0124/token_collapse_27e564a/invariance_short_run/run_invariance_short.py --lambdas 0.0 --max_steps 800 --max_train_samples 2000 --max_val_samples 500 --batch_size 2 --num_workers 2 --use_amp --gradient_accumulation_steps 2`

---

## Step 3: Summary (done)

Summary:
- `summary.json` / `summary.md` 已更新，含 λ=0.0/0.05/0.10。

Next:
- 更新 CONCLUSION Decision。

Blockers:
- None.

Commands / Entrypoints:
- TBD

---

## Step 4: Decision (done)

Summary:
- `CONCLUSION.md` 已補上短跑 Decision（No‑Go）與下一步建議。

Next:
- 無。

Blockers:
- None.

Commands / Entrypoints:
- TBD

---

## Step 5: Global-shift aligned invariance (done)

Summary:
- 完成 `global_shift_k=3` 的短跑（λ=0.05/0.10），輸出 `invariance_short_run_shift/summary.{json,md}`。
- λ=0.05：val strict fw=0.007204；entropy=5.681；top‑k mass=0.306；KL=2.268；token_change_rate=0.9354；p50=0.008885。
- λ=0.10：val strict fw=0.006030；entropy=5.485；top‑k mass=0.311；KL=1.608；token_change_rate=0.9167；p50=0.009547。

Next:
- 更新 CONCLUSION Decision（global‑shift 仍未顯著降低 token_change_rate）。

Blockers:
- 無。

Commands / Entrypoints:
- `source /home/sbplab/miniconda3/etc/profile.d/conda.sh && conda activate test && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python exp_0124/token_collapse_27e564a/invariance_short_run/run_invariance_short.py --output_root exp_0124/token_collapse_27e564a/invariance_short_run_shift/runs --lambdas 0.05,0.10 --global_shift_k 3 --max_steps 800 --max_train_samples 2000 --max_val_samples 500 --batch_size 1 --num_workers 0 --use_amp --gradient_accumulation_steps 2 --log_every 50`

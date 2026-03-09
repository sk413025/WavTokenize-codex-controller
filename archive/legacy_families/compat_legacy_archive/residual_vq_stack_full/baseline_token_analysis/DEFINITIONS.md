# Token Distribution Analysis: Definitions & Pitfalls

This folder contains **REAL token statistics** collected from model outputs (teacher/student) at a specific checkpoint.

## What is REAL vs SIMULATED?

- **REAL**: `real_*_token_ranking.csv`, `real_token_statistics.json`, `FINAL_*` files
  - Produced by `exp_0128/collect_real_tokens.py` + `exp_0128/visualize_complete_analysis.py`
  - Uses **actual token IDs and counts** from running the model.

- **SIMULATED (from metrics)**: outputs under `exp_0128/baseline_token_analysis_from_metrics/`
  - Produced by `exp_0128/analyze_from_existing_metrics.py`
  - Uses collapse metrics (entropy / top-k mass / used codes) to **simulate** frequency patterns.
  - This is **not** the real teacher/student token distribution.

## Units (common confusion)

- In REAL analysis (`real_token_statistics.json`, `FINAL_complete_token_analysis.json`):
  - `top_10_mass`, `top_50_mass`, `top_100_mass` are **percent (%)**.

- In training metrics (`metrics_history.json`, `summary.json` from RVQ runs):
  - `layer0_top10_mass` / `top_10_mass` are typically **fractions (0~1)**.

To avoid mixing units, prefer explicit fields when available:
- `*_pct` for percent, `*_frac` for fraction.

## What does “collapse” mean here?

Practical collapse signals in RVQ experiments:
- `top10_mass → near 1.0` **and**
- `used_codes → very small` (single digits / teens)

REAL teacher distributions can still be naturally non-uniform (e.g., top-10 mass in the 10–30% range) without being “collapsed”.


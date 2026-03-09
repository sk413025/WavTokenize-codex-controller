#!/usr/bin/env python3
"""
Compute statistical tests required by SPEC for commit 5e859b0 analysis.

Methods:
- Spearman correlation
- Mann-Whitney U for bin differences
- Bootstrap CI for means
- FDR (Benjamini-Hochberg) on multiple pairwise tests
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "exp_0217" / "analysis_commit_5e859b0"
QUALITY_JSON = ANALYSIS_DIR / "audio_quality_by_epoch.json"
QUALITY_MD = ANALYSIS_DIR / "audio_quality_by_epoch.md"
STRAT_MD = ANALYSIS_DIR / "stratified_quality_report.md"
HYP_JSON = ANALYSIS_DIR / "hypothesis_scoring.json"
NEXT_MD = ANALYSIS_DIR / "next_experiment_recommendation.md"

OUT_JSON = ANALYSIS_DIR / "statistical_tests_summary.json"
OUT_MD = ANALYSIS_DIR / "statistical_tests_summary.md"

SEED = 42


T453_BINS = [
    (0.0, 0.1, "[0,0.1)"),
    (0.1, 0.2, "[0.1,0.2)"),
    (0.2, 0.3, "[0.2,0.3)"),
    (0.3, 0.5 + 1e-8, "[0.3,0.5]"),
]
SNR_BINS = [
    (-1e9, 0.0, "<0dB"),
    (0.0, 10.0, "0~10dB"),
    (10.0, 20.0, "10~20dB"),
    (20.0, 1e9, ">20dB"),
]
LEN_BINS = [
    (0.0, 2.0, "<2s"),
    (2.0, 5.0, "2~5s"),
    (5.0, 1e9, ">5s"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def collect_rows(quality: dict, split: str, epochs: Optional[Sequence[int]] = None) -> List[dict]:
    rows: List[dict] = []
    if epochs is None:
        keys = sorted(quality["records"][split].keys())
    else:
        keys = [f"epoch_{ep:03d}" for ep in epochs]
    for k in keys:
        rows.extend(quality["records"][split].get(k, []))
    return rows


def vec(rows: List[dict], key: str) -> np.ndarray:
    values = [safe_float(r.get(key)) for r in rows]
    values = [v for v in values if v is not None]
    return np.asarray(values, dtype=np.float64)


def spearman_pair(rows: List[dict], x_key: str, y_key: str) -> dict:
    xs = []
    ys = []
    for r in rows:
        x = safe_float(r.get(x_key))
        y = safe_float(r.get(y_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 3:
        return {"n": len(xs), "rho": None, "p_value": None}
    res = spearmanr(xs, ys)
    return {"n": len(xs), "rho": float(res.statistic), "p_value": float(res.pvalue)}


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = SEED) -> dict:
    if values.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    mean = float(values.mean())
    if values.size == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": mean, "ci_low": lo, "ci_high": hi}


def in_bin(v: float, lo: float, hi: float, hi_closed: bool = False) -> bool:
    if hi_closed:
        return lo <= v <= hi
    return lo <= v < hi


def assign_bin(v: Optional[float], bins: List[Tuple[float, float, str]], hi_closed_last: bool = False) -> Optional[str]:
    if v is None:
        return None
    for i, (lo, hi, label) in enumerate(bins):
        is_last = i == len(bins) - 1
        if in_bin(v, lo, hi, hi_closed=(hi_closed_last and is_last)):
            return label
    return None


def bh_fdr(p_values: List[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.asarray(p_values)[order]
    q_ranked = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = ranked[i] * m / rank
        q = min(q, prev, 1.0)
        q_ranked[i] = q
        prev = q
    q = np.empty(m, dtype=np.float64)
    q[order] = q_ranked
    return q.tolist()


def build_bin_groups(rows: List[dict], value_key: str, bins: List[Tuple[float, float, str]], hi_closed_last: bool) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {label: [] for _, _, label in bins}
    for r in rows:
        v = safe_float(r.get(value_key))
        label = assign_bin(v, bins, hi_closed_last=hi_closed_last)
        if label is not None:
            out[label].append(r)
    return out


def pairwise_mw(groups: Dict[str, List[dict]], metric_key: str, min_n: int = 5) -> List[dict]:
    labels = list(groups.keys())
    tests: List[dict] = []
    for a, b in itertools.combinations(labels, 2):
        va = np.asarray([safe_float(r.get(metric_key)) for r in groups[a]], dtype=np.float64)
        vb = np.asarray([safe_float(r.get(metric_key)) for r in groups[b]], dtype=np.float64)
        va = va[np.isfinite(va)]
        vb = vb[np.isfinite(vb)]
        row = {"bin_a": a, "bin_b": b, "metric": metric_key, "n_a": int(va.size), "n_b": int(vb.size)}
        if va.size < min_n or vb.size < min_n:
            row.update({"status": "insufficient_n", "u_stat": None, "p_value": None})
        else:
            res = mannwhitneyu(va, vb, alternative="two-sided")
            row.update({"status": "ok", "u_stat": float(res.statistic), "p_value": float(res.pvalue)})
        tests.append(row)
    return tests


def build_bin_summary(groups: Dict[str, List[dict]], metric_keys: Sequence[str], seed_offset: int = 0) -> Dict[str, dict]:
    out = {}
    for label, rows in groups.items():
        out[label] = {"n": len(rows)}
        for j, mk in enumerate(metric_keys):
            vals = np.asarray([safe_float(r.get(mk)) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out[label][mk] = bootstrap_mean_ci(vals, seed=SEED + seed_offset + j * 97)
    return out


def append_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    if section.strip() in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + section.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def update_hypothesis_blockers(has_stats: bool) -> None:
    data = load_json(HYP_JSON)
    blockers = data.get("blocked_by", [])
    keep = []
    for b in blockers:
        if "Mann-Whitney" in b or "Bootstrap" in b or "FDR" in b or "statistical tests" in b:
            continue
        keep.append(b)
    if not has_stats:
        keep.append("Mann-Whitney/Bootstrap + FDR statistical tests not yet executed")
    data["blocked_by"] = keep
    dump_json(HYP_JSON, data)


def update_next_recommendation(trigger_b_met: bool) -> None:
    text = NEXT_MD.read_text(encoding="utf-8")
    old = "- Trigger B: objective mismatch remains significant after statistical test (`p < 0.05`).\n  - **Status: pending**"
    new = "- Trigger B: objective mismatch remains significant after statistical test (`p < 0.05`).\n  - **Status: met**"
    if trigger_b_met and old in text:
        text = text.replace(old, new)
    NEXT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    quality = load_json(QUALITY_JSON)

    # 1) Spearman correlations
    train_rows = collect_rows(quality, "train")
    val_rows = collect_rows(quality, "val")
    val300_rows = collect_rows(quality, "val", epochs=[300])

    correlations = {
        "train_all_epochs": {
            "feature_mse_vs_delta_pesq": spearman_pair(train_rows, "feature_mse", "delta_pesq"),
            "feature_mse_vs_delta_stoi": spearman_pair(train_rows, "feature_mse", "delta_stoi"),
            "t453_ratio_vs_delta_pesq": spearman_pair(train_rows, "t453_ratio", "delta_pesq"),
            "t453_ratio_vs_delta_stoi": spearman_pair(train_rows, "t453_ratio", "delta_stoi"),
        },
        "val_all_epochs": {
            "feature_mse_vs_delta_pesq": spearman_pair(val_rows, "feature_mse", "delta_pesq"),
            "feature_mse_vs_delta_stoi": spearman_pair(val_rows, "feature_mse", "delta_stoi"),
            "t453_ratio_vs_delta_pesq": spearman_pair(val_rows, "t453_ratio", "delta_pesq"),
            "t453_ratio_vs_delta_stoi": spearman_pair(val_rows, "t453_ratio", "delta_stoi"),
        },
        "val_epoch300": {
            "feature_mse_vs_delta_pesq": spearman_pair(val300_rows, "feature_mse", "delta_pesq"),
            "feature_mse_vs_delta_stoi": spearman_pair(val300_rows, "feature_mse", "delta_stoi"),
            "t453_ratio_vs_delta_pesq": spearman_pair(val300_rows, "t453_ratio", "delta_pesq"),
            "t453_ratio_vs_delta_stoi": spearman_pair(val300_rows, "t453_ratio", "delta_stoi"),
        },
    }

    # 2) Bootstrap CI for objective mismatch by epoch (val)
    val_epoch_bootstrap = {}
    for ep_key, rows in sorted(quality["records"]["val"].items()):
        if not rows:
            continue
        d_p = vec(rows, "delta_pesq")
        d_s = vec(rows, "delta_stoi")
        val_epoch_bootstrap[ep_key] = {
            "n": int(len(rows)),
            "delta_pesq": bootstrap_mean_ci(d_p, seed=SEED + int(ep_key[-3:])),
            "delta_stoi": bootstrap_mean_ci(d_s, seed=SEED + 1000 + int(ep_key[-3:])),
        }

    # 3) Bin-based tests on val@300
    t453_groups = build_bin_groups(val300_rows, "t453_ratio", T453_BINS, hi_closed_last=True)
    snr_groups = build_bin_groups(val300_rows, "snr_db", SNR_BINS, hi_closed_last=False)
    len_groups = build_bin_groups(val300_rows, "duration_sec", LEN_BINS, hi_closed_last=False)

    pairwise_tests = {
        "t453_ratio": pairwise_mw(t453_groups, "delta_pesq") + pairwise_mw(t453_groups, "delta_stoi"),
        "snr_db": pairwise_mw(snr_groups, "delta_pesq") + pairwise_mw(snr_groups, "delta_stoi"),
        "duration_sec": pairwise_mw(len_groups, "delta_pesq") + pairwise_mw(len_groups, "delta_stoi"),
    }

    # 4) FDR correction across all valid pairwise tests
    flat = []
    for dim, tests in pairwise_tests.items():
        for i, t in enumerate(tests):
            if t["status"] == "ok" and t["p_value"] is not None:
                flat.append((dim, i, float(t["p_value"])))
    q_vals = bh_fdr([p for _, _, p in flat])
    for (dim, idx, _), q in zip(flat, q_vals):
        pairwise_tests[dim][idx]["q_value_fdr"] = float(q)
        pairwise_tests[dim][idx]["significant_fdr_0p05"] = bool(q < 0.05)
    for dim, tests in pairwise_tests.items():
        for t in tests:
            if "q_value_fdr" not in t:
                t["q_value_fdr"] = None
                t["significant_fdr_0p05"] = False

    # 5) Bootstrap summaries per bin
    bin_summaries = {
        "t453_ratio": build_bin_summary(t453_groups, ["delta_pesq", "delta_stoi", "feature_mse"], seed_offset=200),
        "snr_db": build_bin_summary(snr_groups, ["delta_pesq", "delta_stoi", "feature_mse"], seed_offset=400),
        "duration_sec": build_bin_summary(len_groups, ["delta_pesq", "delta_stoi", "feature_mse"], seed_offset=600),
    }

    results = {
        "scope": "families/deps/encoder_aug/runs/augmented_long_20260216",
        "source": "families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_by_epoch.json",
        "methods": {
            "correlation": "Spearman",
            "bin_difference": "Mann-Whitney U",
            "ci": "Bootstrap percentile 95%",
            "multiple_comparison": "Benjamini-Hochberg FDR",
        },
        "sample_sizes": {
            "train_all_epochs": len(train_rows),
            "val_all_epochs": len(val_rows),
            "val_epoch300": len(val300_rows),
        },
        "correlations": correlations,
        "val_epoch_bootstrap": val_epoch_bootstrap,
        "bin_summaries_val_epoch300": bin_summaries,
        "pairwise_mann_whitney_val_epoch300": pairwise_tests,
    }

    dump_json(OUT_JSON, results)

    # Also attach short pointer into audio_quality_by_epoch.json
    quality["statistical_tests_file"] = str(OUT_JSON.relative_to(ROOT))
    quality["statistical_tests_complete"] = True
    dump_json(QUALITY_JSON, quality)

    # Build markdown summary
    md_lines = []
    md_lines.append("# Statistical Tests Summary (Commit 5e859b0)")
    md_lines.append("")
    md_lines.append(f"- source: `{results['source']}`")
    md_lines.append(f"- n(train_all_epochs): {results['sample_sizes']['train_all_epochs']}")
    md_lines.append(f"- n(val_all_epochs): {results['sample_sizes']['val_all_epochs']}")
    md_lines.append(f"- n(val_epoch300): {results['sample_sizes']['val_epoch300']}")
    md_lines.append("")
    md_lines.append("## Spearman Correlations")
    for scope, corr in correlations.items():
        md_lines.append(f"### {scope}")
        for name, row in corr.items():
            md_lines.append(f"- {name}: rho={fmt(row['rho'], 4)}, p={fmt(row['p_value'], 6)}, n={row['n']}")
        md_lines.append("")

    md_lines.append("## Val Epoch Bootstrap (ΔPESQ / ΔSTOI)")
    md_lines.append("| epoch | n | ΔPESQ mean | 95% CI | ΔSTOI mean | 95% CI |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for ep, row in sorted(val_epoch_bootstrap.items()):
        dp = row["delta_pesq"]
        ds = row["delta_stoi"]
        md_lines.append(
            f"| {ep} | {row['n']} | {fmt(dp['mean'],4)} | [{fmt(dp['ci_low'],4)}, {fmt(dp['ci_high'],4)}] | "
            f"{fmt(ds['mean'],4)} | [{fmt(ds['ci_low'],4)}, {fmt(ds['ci_high'],4)}] |"
        )
    md_lines.append("")

    md_lines.append("## Pairwise Mann-Whitney + FDR (val@epoch300)")
    md_lines.append("| dimension | metric | bin_a | bin_b | n_a | n_b | p | q_fdr | sig(q<0.05) |")
    md_lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for dim, tests in pairwise_tests.items():
        for t in tests:
            if t["status"] != "ok":
                continue
            md_lines.append(
                f"| {dim} | {t['metric']} | {t['bin_a']} | {t['bin_b']} | {t['n_a']} | {t['n_b']} | "
                f"{fmt(t['p_value'],6)} | {fmt(t['q_value_fdr'],6)} | {t['significant_fdr_0p05']} |"
            )
    md_lines.append("")
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    # Append short pointers to existing md files for traceability.
    append_section(
        QUALITY_MD,
        "## Statistical Tests Reference\n"
        f"- Detailed tests: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/{OUT_JSON.name}`\n"
        f"- Human summary: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/{OUT_MD.name}`",
    )
    append_section(
        STRAT_MD,
        "## Statistical Tests Reference\n"
        f"- Pairwise Mann-Whitney + FDR: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/{OUT_MD.name}`",
    )

    # Determine Trigger B status from val_epoch300 feature_mse-vs-delta_pesq p-value.
    p_trigger = correlations["val_epoch300"]["feature_mse_vs_delta_pesq"]["p_value"]
    trigger_b_met = p_trigger is not None and p_trigger < 0.05

    update_hypothesis_blockers(has_stats=True)
    update_next_recommendation(trigger_b_met=trigger_b_met)

    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")
    print(f"Trigger B met: {trigger_b_met}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


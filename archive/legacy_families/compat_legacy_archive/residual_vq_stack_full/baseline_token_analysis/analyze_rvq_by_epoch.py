"""
Analyze RVQ short-run outputs (metrics_history.json) as "epoch" summaries.

This is intended for Scheme B when epoch0/epoch1 checkpoints are missing.
It converts step-based eval points into epoch_end points using steps_per_epoch,
and exports consistent, unit-explicit metrics for:
  - val_student (RVQ layer0)
  - val_teacher (baseline single VQ; reference only)
  - train_student
  - train_teacher

Outputs (written to run_dir unless specified otherwise):
  - epoch_metrics.json
  - epoch_metrics.csv
  - delta_epoch0_epoch1.json
  - (optional) trend plots under out_trends_dir
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _pct_from_frac(frac: Any) -> float | None:
    v = _to_float(frac)
    if v is None:
        return None
    return 100.0 * v


@dataclass(frozen=True)
class SplitSpec:
    name: str
    prefix: str
    kind: str  # "student" or "teacher"


_SPLITS: list[SplitSpec] = [
    SplitSpec(name="val_student", prefix="", kind="student"),
    SplitSpec(name="val_teacher", prefix="", kind="teacher"),
    SplitSpec(name="train_student", prefix="train_", kind="student"),
    SplitSpec(name="train_teacher", prefix="train_", kind="teacher"),
]


def _extract_student(record: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    ent = _to_float(record.get(f"{prefix}layer0_entropy"))
    used = _to_int(record.get(f"{prefix}layer0_used_codes"))
    total = _to_int(record.get(f"{prefix}layer0_total_codes"))
    usage_pct = _to_float(record.get(f"{prefix}layer0_usage_pct"))

    # top-k mass (%): prefer explicit *_pct keys; otherwise convert from fraction.
    top1_pct = _to_float(record.get(f"{prefix}layer0_top1_mass_pct"))
    if top1_pct is None:
        top1_pct = _pct_from_frac(record.get(f"{prefix}layer0_top1_mass"))

    top10_pct = _to_float(record.get(f"{prefix}layer0_top10_mass_pct"))
    if top10_pct is None:
        top10_pct = _pct_from_frac(record.get(f"{prefix}layer0_top10_mass"))

    top50_pct = _to_float(record.get(f"{prefix}layer0_top50_mass_pct"))
    if top50_pct is None:
        top50_pct = _pct_from_frac(record.get(f"{prefix}layer0_top50_mass"))

    top100_pct = _to_float(record.get(f"{prefix}layer0_top100_mass_pct"))
    if top100_pct is None:
        top100_pct = _pct_from_frac(record.get(f"{prefix}layer0_top100_mass"))

    feature_mse = _to_float(record.get(f"{prefix}feature_mse"))

    if ent is None and used is None and top10_pct is None:
        return None

    if usage_pct is None and used is not None and total:
        usage_pct = 100.0 * float(used) / float(total)

    return {
        "entropy_bits": ent,
        "top_1_mass_pct": top1_pct,
        "top_10_mass_pct": top10_pct,
        "top_50_mass_pct": top50_pct,
        "top_100_mass_pct": top100_pct,
        "used_codes": used,
        "total_codes": total,
        "usage_pct": usage_pct,
        "feature_mse": feature_mse,
    }


def _extract_teacher(record: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    ent = _to_float(record.get(f"{prefix}teacher_entropy"))
    used = _to_int(record.get(f"{prefix}teacher_used_codes"))
    total = _to_int(record.get(f"{prefix}teacher_total_codes"))
    usage_pct = _to_float(record.get(f"{prefix}teacher_usage_pct"))

    top1_pct = _to_float(record.get(f"{prefix}teacher_top1_mass_pct"))
    if top1_pct is None:
        top1_pct = _pct_from_frac(record.get(f"{prefix}teacher_top1_mass"))

    top10_pct = _to_float(record.get(f"{prefix}teacher_top10_mass_pct"))
    if top10_pct is None:
        top10_pct = _pct_from_frac(record.get(f"{prefix}teacher_top10_mass"))

    top50_pct = _to_float(record.get(f"{prefix}teacher_top50_mass_pct"))
    if top50_pct is None:
        top50_pct = _pct_from_frac(record.get(f"{prefix}teacher_top50_mass"))

    top100_pct = _to_float(record.get(f"{prefix}teacher_top100_mass_pct"))
    if top100_pct is None:
        top100_pct = _pct_from_frac(record.get(f"{prefix}teacher_top100_mass"))

    if ent is None and used is None and top10_pct is None:
        return None

    if usage_pct is None and used is not None and total:
        usage_pct = 100.0 * float(used) / float(total)

    return {
        "entropy_bits": ent,
        "top_1_mass_pct": top1_pct,
        "top_10_mass_pct": top10_pct,
        "top_50_mass_pct": top50_pct,
        "top_100_mass_pct": top100_pct,
        "used_codes": used,
        "total_codes": total,
        "usage_pct": usage_pct,
        "feature_mse": None,
    }


def _extract_split(record: dict[str, Any], split: SplitSpec) -> dict[str, Any] | None:
    if split.kind == "student":
        return _extract_student(record, split.prefix)
    if split.kind == "teacher":
        return _extract_teacher(record, split.prefix)
    raise ValueError(f"Unknown split kind: {split.kind}")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = [
        "epoch_end",
        "step",
        "split",
        "entropy_bits",
        "top_1_mass_pct",
        "top_10_mass_pct",
        "top_50_mass_pct",
        "top_100_mass_pct",
        "used_codes",
        "total_codes",
        "usage_pct",
        "feature_mse",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _plot_trends(rows: list[dict[str, Any]], out_dir: Path, title_prefix: str) -> None:
    # Lazy import: matplotlib is optional for JSON/CSV export.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build series for student splits only.
    def series(split: str, key: str) -> tuple[list[int], list[float]]:
        pts = [(int(r["epoch_end"]), r.get(key)) for r in rows if r["split"] == split]
        pts = [(e, v) for e, v in pts if v is not None]
        pts.sort(key=lambda x: x[0])
        return [e for e, _ in pts], [float(v) for _, v in pts]

    student_splits = ["val_student", "train_student"]
    plots = [
        ("entropy_bits", "Entropy (bits)", "student_layer0_entropy_vs_epoch.png"),
        ("top_10_mass_pct", "Top-10 mass (%)", "student_layer0_top10_mass_pct_vs_epoch.png"),
        ("used_codes", "Used codes", "student_layer0_used_codes_vs_epoch.png"),
        ("usage_pct", "Usage (%)", "student_layer0_usage_pct_vs_epoch.png"),
        ("feature_mse", "Feature MSE", "student_feature_mse_vs_epoch.png"),
    ]

    for key, ylabel, fname in plots:
        plt.figure(figsize=(7.5, 4.5))
        for split in student_splits:
            xs, ys = series(split, key)
            if xs:
                plt.plot(xs, ys, marker="o", linewidth=2, label=split)
        plt.xlabel("epoch_end")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} {key}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=160)
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Directory containing metrics_history.json")
    p.add_argument("--steps_per_epoch", type=int, default=486)
    p.add_argument("--out_trends_dir", type=str, default="", help="If set, write plots to this directory")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics_history.json"
    if not metrics_path.exists():
        raise FileNotFoundError(str(metrics_path))

    metrics_history: list[dict[str, Any]] = _load_json(metrics_path)
    metrics_history = sorted(metrics_history, key=lambda r: int(r.get("step", 0)))

    steps_per_epoch = int(args.steps_per_epoch)
    epoch_rows: list[dict[str, Any]] = []

    # Use only epoch-boundary points for stable comparison.
    for rec in metrics_history:
        step = _to_int(rec.get("step"))
        if step is None:
            continue
        if step % steps_per_epoch != 0:
            continue

        epoch_end = step // steps_per_epoch

        for split in _SPLITS:
            payload = _extract_split(rec, split)
            if payload is None:
                continue
            epoch_rows.append(
                {
                    "epoch_end": int(epoch_end),
                    "step": int(step),
                    "split": split.name,
                    **payload,
                }
            )

    out_json = run_dir / "epoch_metrics.json"
    out_csv = run_dir / "epoch_metrics.csv"
    _write_json(
        out_json,
        {
            "run_dir": str(run_dir),
            "metrics_history": str(metrics_path),
            "steps_per_epoch": steps_per_epoch,
            "rows": epoch_rows,
        },
    )
    _write_csv(out_csv, epoch_rows)

    # Delta epoch0->epoch1
    def get_row(epoch_end: int, split: str) -> dict[str, Any] | None:
        for r in epoch_rows:
            if int(r["epoch_end"]) == int(epoch_end) and r["split"] == split:
                return r
        return None

    deltas: dict[str, Any] = {"steps_per_epoch": steps_per_epoch, "epoch0": {}, "epoch1": {}, "delta": {}}
    for split in [s.name for s in _SPLITS]:
        r0 = get_row(0, split)
        r1 = get_row(1, split)
        if r0 is None or r1 is None:
            continue

        keys = ["entropy_bits", "top_10_mass_pct", "used_codes", "usage_pct", "feature_mse"]
        deltas["epoch0"][split] = {k: r0.get(k) for k in keys}
        deltas["epoch1"][split] = {k: r1.get(k) for k in keys}
        deltas["delta"][split] = {
            k: (None if r0.get(k) is None or r1.get(k) is None else float(r1[k]) - float(r0[k]))
            for k in keys
        }

    out_delta = run_dir / "delta_epoch0_epoch1.json"
    _write_json(out_delta, deltas)

    if args.out_trends_dir:
        _plot_trends(epoch_rows, Path(args.out_trends_dir), title_prefix=run_dir.name)

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_delta}")
    if args.out_trends_dir:
        print(f"[OK] wrote plots to: {args.out_trends_dir}")


if __name__ == "__main__":
    main()


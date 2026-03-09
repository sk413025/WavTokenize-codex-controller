#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROLE_CSV = "/home/sbplab/ruizi/WavTokenize-feature-analysis/families/official/material_generalization/wavtokenizer_featuremap_14wav_extended/conv18_14wav_role_metrics.csv"
DEFAULT_OUTDIR = "/home/sbplab/ruizi/WavTokenize-feature-analysis/families/official/material_generalization/wavtokenizer_featuremap_14wav_extended"


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["layer_idx"] = int(row["layer_idx"])
            for k in [
                "noise_sensitivity",
                "speaker_identity_score",
                "content_shared_score",
                "temporal_detail_score_norm",
            ]:
                row[k] = float(row[k])
            rows.append(row)
    rows.sort(key=lambda x: x["layer_idx"])
    return rows


def score_layer(row):
    # Higher adapt_score -> better LoRA candidate for denoising.
    adapt_score = (
        0.70 * row["noise_sensitivity"]
        + 0.20 * row["temporal_detail_score_norm"]
        + 0.10 * (1.0 - row["content_shared_score"])
    )
    # Higher freeze_score -> better keep-frozen candidate for preserving clarity/content.
    freeze_score = (
        0.70 * row["content_shared_score"]
        + 0.20 * row["speaker_identity_score"]
        + 0.10 * (1.0 - row["noise_sensitivity"])
    )
    margin = adapt_score - freeze_score
    return adapt_score, freeze_score, margin


def decide(margin: float, thr: float = 0.08) -> str:
    if margin >= thr:
        return "MOVE"
    if margin <= -thr:
        return "FREEZE"
    return "UNCERTAIN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role_csv", default=DEFAULT_ROLE_CSV)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--threshold", type=float, default=0.08)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.role_csv)
    for row in rows:
        adapt, freeze, margin = score_layer(row)
        row["adapt_score"] = adapt
        row["freeze_score"] = freeze
        row["decision_margin"] = margin
        row["decision"] = decide(margin, args.threshold)

    sorted_rows = sorted(rows, key=lambda x: x["decision_margin"], reverse=True)

    # Plot one direct evidence figure: sorted decision margin bar chart
    labels = [f"L{r['layer_idx']:02d}" for r in sorted_rows]
    vals = [r["decision_margin"] for r in sorted_rows]
    colors = []
    for r in sorted_rows:
        if r["decision"] == "MOVE":
            colors.append("#d62728")   # red
        elif r["decision"] == "FREEZE":
            colors.append("#1f77b4")   # blue
        else:
            colors.append("#7f7f7f")   # gray

    plt.figure(figsize=(14, 6))
    x = np.arange(len(sorted_rows))
    plt.bar(x, vals, color=colors, alpha=0.9)
    plt.axhline(0.0, color="black", linewidth=1.2)
    plt.axhline(args.threshold, color="#d62728", linestyle="--", linewidth=1.0, label=f"MOVE threshold +{args.threshold:.2f}")
    plt.axhline(-args.threshold, color="#1f77b4", linestyle="--", linewidth=1.0, label=f"FREEZE threshold -{args.threshold:.2f}")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Decision Margin = Adapt Score - Freeze Score")
    plt.xlabel("Layer (sorted by margin)")
    plt.title("Direct LoRA Layer Decision Evidence (14-file analysis)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig_path = outdir / "lora_layer_decision_evidence.png"
    plt.savefig(fig_path, dpi=240)
    plt.close()

    # Save detailed evidence table
    evidence_csv = outdir / "lora_layer_decision_evidence.csv"
    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layer_idx",
            "module_path",
            "stage",
            "noise_sensitivity",
            "speaker_identity_score",
            "content_shared_score",
            "temporal_detail_score_norm",
            "adapt_score",
            "freeze_score",
            "decision_margin",
            "decision",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted_rows:
            w.writerow({k: r[k] for k in fieldnames})

    summary = {
        "threshold": args.threshold,
        "move_layers": [int(r["layer_idx"]) for r in sorted_rows if r["decision"] == "MOVE"],
        "freeze_layers": [int(r["layer_idx"]) for r in sorted_rows if r["decision"] == "FREEZE"],
        "uncertain_layers": [int(r["layer_idx"]) for r in sorted_rows if r["decision"] == "UNCERTAIN"],
        "top5_move": [
            {
                "layer_idx": int(r["layer_idx"]),
                "module_path": r["module_path"],
                "decision_margin": float(r["decision_margin"]),
            }
            for r in sorted_rows[:5]
        ],
        "top5_freeze": [
            {
                "layer_idx": int(r["layer_idx"]),
                "module_path": r["module_path"],
                "decision_margin": float(r["decision_margin"]),
            }
            for r in sorted_rows[-5:]
        ],
    }
    summary_path = outdir / "lora_layer_decision_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] {fig_path}")
    print(f"[OK] {evidence_csv}")
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()

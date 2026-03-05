#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUMMARY = "/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0304/wavtokenizer_featuremap_6wav/analysis_summary.json"
DEFAULT_OUTDIR = "/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0304/wavtokenizer_featuremap_6wav"


def short_layer_name(desc: str) -> str:
    # "03: SConv1d downsample (k=4, s=2, 32->64)" -> "03 SConv1d s2"
    prefix, rest = desc.split(": ", 1)
    if "SConv1d downsample" in rest:
        # get stride info if present
        s = "s?"
        if "s=" in rest:
            s = "s=" + rest.split("s=")[1].split(",")[0].replace(")", "")
        return f"{prefix} Conv {s}"
    if "SConv1d conv" in rest:
        return f"{prefix} Conv"
    if "SEANetResnetBlock" in rest:
        return f"{prefix} ResBlk"
    if "SLSTM" in rest:
        return f"{prefix} LSTM"
    if "ELU" in rest:
        return f"{prefix} ELU"
    return f"{prefix} {rest.split()[0]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", default=DEFAULT_SUMMARY)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data["encoder_structure"]["layers"]
    x = list(range(len(layers)))
    x_labels = [short_layer_name(item["desc"]) for item in layers]

    pair_data = data["clean_vs_ldv_pairs"]
    legend_map = {
        "boy4_clean_vs_ldv": "boy4 clean vs LDV",
        "boy7_clean_vs_ldv": "boy7 clean vs LDV",
        "girl9_clean_vs_ldv": "girl9 clean vs LDV",
    }

    # Build Y values in layer index order
    lines = {}
    for pair_key, pair_value in pair_data.items():
        lw = pair_value["layerwise_cosine"]
        y = []
        for i, layer in enumerate(layers):
            k = f"{i:02d}_{layer['type']}"
            y.append(float(lw[k]))
        lines[pair_key] = y

    # Line chart
    plt.figure(figsize=(15, 5))
    styles = {
        "boy4_clean_vs_ldv": ("#1f77b4", "o"),
        "boy7_clean_vs_ldv": ("#d62728", "s"),
        "girl9_clean_vs_ldv": ("#2ca02c", "^"),
    }
    for key, y in lines.items():
        color, marker = styles.get(key, ("#333333", "o"))
        plt.plot(x, y, color=color, marker=marker, linewidth=2, markersize=5, label=legend_map.get(key, key))

    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.xticks(x, x_labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Cosine Similarity")
    plt.xlabel("Encoder Layer")
    plt.title("WavTokenizer Layerwise Feature Similarity (Clean vs LDV)")
    plt.legend()
    plt.tight_layout()
    line_png = outdir / "layerwise_clean_vs_ldv_cosine_line.png"
    plt.savefig(line_png, dpi=220)
    plt.close()

    # Heatmap
    order = ["boy4_clean_vs_ldv", "boy7_clean_vs_ldv", "girl9_clean_vs_ldv"]
    heat = np.array([lines[k] for k in order], dtype=np.float32)
    row_labels = [legend_map[k] for k in order]

    plt.figure(figsize=(15, 3.8))
    im = plt.imshow(heat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, fraction=0.03, pad=0.02, label="Cosine")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xticks(x, x_labels, rotation=45, ha="right", fontsize=9)
    plt.title("WavTokenizer Layerwise Similarity Heatmap (Clean vs LDV)")
    plt.tight_layout()
    heat_png = outdir / "layerwise_clean_vs_ldv_cosine_heatmap.png"
    plt.savefig(heat_png, dpi=220)
    plt.close()

    # Text summary for quick reading
    lines_txt = []
    for pair_key in order:
        vals = np.array(lines[pair_key], dtype=np.float32)
        min_idx = int(vals.argmin())
        max_idx = int(vals.argmax())
        lines_txt.append(
            f"{legend_map[pair_key]}: min={vals[min_idx]:.4f} @ layer {min_idx:02d}, "
            f"max={vals[max_idx]:.4f} @ layer {max_idx:02d}, mean={vals.mean():.4f}"
        )
    (outdir / "layerwise_plot_summary.txt").write_text("\n".join(lines_txt) + "\n", encoding="utf-8")

    print(f"[OK] {line_png}")
    print(f"[OK] {heat_png}")
    print(f"[OK] {outdir / 'layerwise_plot_summary.txt'}")


if __name__ == "__main__":
    main()

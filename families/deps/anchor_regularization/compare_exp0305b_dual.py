#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunSummary:
    name: str
    run_dir: Path
    epochs: int
    anchor_layers: List[int]
    lambda_anchor: float
    best_epoch: int
    best_val_wav_mse: float
    best_val_noisy_mse: float
    best_val_anchor: float
    best_train_anchor: float
    best_improve_pct: float
    final_val_wav_mse: float
    final_val_anchor: float


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_summary(name: str, run_dir: Path) -> RunSummary:
    cfg = load_json(run_dir / "config.json")
    hist = load_json(run_dir / "history.json")

    val_wav = hist["val_wav_mse"]
    val_noisy = hist["val_noisy_mse"]
    val_anchor = hist["val_anchor"]
    train_anchor = hist["train_anchor"]

    best_idx = int(np.argmin(val_wav))
    best_val_wav = float(val_wav[best_idx])
    best_noisy = float(val_noisy[best_idx])
    improve = (best_noisy - best_val_wav) / (best_noisy + 1e-9) * 100.0

    return RunSummary(
        name=name,
        run_dir=run_dir,
        epochs=len(val_wav),
        anchor_layers=cfg.get("anchor_layer_ids", []),
        lambda_anchor=float(cfg.get("lambda_anchor", float("nan"))),
        best_epoch=best_idx + 1,
        best_val_wav_mse=best_val_wav,
        best_val_noisy_mse=best_noisy,
        best_val_anchor=float(val_anchor[best_idx]),
        best_train_anchor=float(train_anchor[best_idx]),
        best_improve_pct=float(improve),
        final_val_wav_mse=float(val_wav[-1]),
        final_val_anchor=float(val_anchor[-1]),
    )


def save_table(rows: List[RunSummary], out_csv: Path):
    fields = [
        "name",
        "run_dir",
        "epochs",
        "anchor_layers",
        "lambda_anchor",
        "best_epoch",
        "best_val_wav_mse",
        "best_val_noisy_mse",
        "best_improve_pct",
        "best_val_anchor",
        "best_train_anchor",
        "final_val_wav_mse",
        "final_val_anchor",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "name": r.name,
                "run_dir": str(r.run_dir),
                "epochs": r.epochs,
                "anchor_layers": ",".join(str(x) for x in r.anchor_layers),
                "lambda_anchor": r.lambda_anchor,
                "best_epoch": r.best_epoch,
                "best_val_wav_mse": r.best_val_wav_mse,
                "best_val_noisy_mse": r.best_val_noisy_mse,
                "best_improve_pct": r.best_improve_pct,
                "best_val_anchor": r.best_val_anchor,
                "best_train_anchor": r.best_train_anchor,
                "final_val_wav_mse": r.final_val_wav_mse,
                "final_val_anchor": r.final_val_anchor,
            })


def plot_curves(tail_dir: Path, front_dir: Path, out_png: Path):
    tail_hist = load_json(tail_dir / "history.json")
    front_hist = load_json(front_dir / "history.json")

    t_ep = np.arange(1, len(tail_hist["val_wav_mse"]) + 1)
    f_ep = np.arange(1, len(front_hist["val_wav_mse"]) + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    axes[0].plot(t_ep, tail_hist["val_wav_mse"], "o-", label="tail_lock val_wav_mse")
    axes[0].plot(f_ep, front_hist["val_wav_mse"], "s-", label="front_tail_lock val_wav_mse")
    axes[0].plot(t_ep, tail_hist["val_noisy_mse"], "--", alpha=0.7, label="tail_lock noisy baseline")
    axes[0].plot(f_ep, front_hist["val_noisy_mse"], "--", alpha=0.7, label="front_tail_lock noisy baseline")
    axes[0].set_title("Val MSE Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(t_ep, tail_hist["val_anchor"], "o-", label="tail_lock val_anchor")
    axes[1].plot(f_ep, front_hist["val_anchor"], "s-", label="front_tail_lock val_anchor")
    axes[1].plot(t_ep, tail_hist["train_anchor"], "--", alpha=0.8, label="tail_lock train_anchor")
    axes[1].plot(f_ep, front_hist["train_anchor"], "--", alpha=0.8, label="front_tail_lock train_anchor")
    axes[1].set_title("Anchor Loss Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Anchor MSE")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def save_milestone_table(
    tail_dir: Path,
    front_dir: Path,
    out_csv: Path,
    interval: int,
):
    tail_hist = load_json(tail_dir / "history.json")
    front_hist = load_json(front_dir / "history.json")

    max_ep = min(len(tail_hist["val_wav_mse"]), len(front_hist["val_wav_mse"]))
    if max_ep <= 0:
        return

    milestones = list(range(interval, max_ep + 1, interval))
    if max_ep not in milestones:
        milestones.append(max_ep)

    fields = [
        "epoch",
        "tail_val_wav_mse",
        "tail_val_noisy_mse",
        "tail_improve_pct",
        "tail_val_anchor",
        "front_val_wav_mse",
        "front_val_noisy_mse",
        "front_improve_pct",
        "front_val_anchor",
        "winner_by_val_wav_mse",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ep in milestones:
            i = ep - 1
            tw = float(tail_hist["val_wav_mse"][i])
            tn = float(tail_hist["val_noisy_mse"][i])
            ta = float(tail_hist["val_anchor"][i])
            fw = float(front_hist["val_wav_mse"][i])
            fn = float(front_hist["val_noisy_mse"][i])
            fa = float(front_hist["val_anchor"][i])
            t_imp = (tn - tw) / (tn + 1e-9) * 100.0
            f_imp = (fn - fw) / (fn + 1e-9) * 100.0
            winner = "tail_lock_L16L17" if tw < fw else "front_tail_lock_L0L1L16L17"
            w.writerow({
                "epoch": ep,
                "tail_val_wav_mse": tw,
                "tail_val_noisy_mse": tn,
                "tail_improve_pct": t_imp,
                "tail_val_anchor": ta,
                "front_val_wav_mse": fw,
                "front_val_noisy_mse": fn,
                "front_improve_pct": f_imp,
                "front_val_anchor": fa,
                "winner_by_val_wav_mse": winner,
            })


def print_console(rows: List[RunSummary]):
    print("=" * 88)
    print("exp_0305b comparison")
    print("=" * 88)
    for r in rows:
        print(
            f"[{r.name}] layers={r.anchor_layers} lambda_anchor={r.lambda_anchor} "
            f"best_epoch={r.best_epoch} best_val_wav_mse={r.best_val_wav_mse:.6f} "
            f"improve={r.best_improve_pct:+.2f}% best_val_anchor={r.best_val_anchor:.6f}"
        )
    winner = min(rows, key=lambda x: x.best_val_wav_mse)
    print("-" * 88)
    print(f"Winner by best_val_wav_mse: {winner.name}")
    print("=" * 88)


def main():
    p = argparse.ArgumentParser(description="Compare exp_0305b tail_lock vs front_tail_lock runs")
    p.add_argument("--tail_dir", required=True, type=Path)
    p.add_argument("--front_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--milestone_interval", type=int, default=50,
                   help="Save milestone comparison every N epochs (default: 50)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tail = build_summary("tail_lock_L16L17", args.tail_dir)
    front = build_summary("front_tail_lock_L0L1L16L17", args.front_dir)
    rows = [tail, front]

    save_table(rows, args.output_dir / "comparison_summary.csv")
    with (args.output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **r.__dict__,
                    "run_dir": str(r.run_dir),
                }
                for r in rows
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    plot_curves(args.tail_dir, args.front_dir, args.output_dir / "comparison_curves.png")
    save_milestone_table(
        args.tail_dir,
        args.front_dir,
        args.output_dir / f"comparison_milestones_every{args.milestone_interval}.csv",
        interval=max(1, args.milestone_interval),
    )
    print_console(rows)


if __name__ == "__main__":
    main()

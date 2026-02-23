#!/usr/bin/env python3
"""
Read-only progress monitor for M1 run.

Generates a structured snapshot from:
- run config
- metrics_history.json (if present)
- train.log progress/summary/warnings
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_RUN = Path("exp_0217/runs/t453_m1_minw03_epoch100_debug")
DEFAULT_OUT = Path("exp_0217/analysis_commit_5e859b0/m1_progress_snapshot.json")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_log(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {
            "exists": False,
            "latest_progress_bar": None,
            "latest_epoch_summary": None,
            "latest_gate_line": None,
            "warnings": {},
        }

    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")

    # Example: Epoch 2:  36%|...| 464/1296
    p_prog = re.compile(r"Epoch\s+(\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")
    prog_matches = list(p_prog.finditer(text))
    latest_prog = None
    if prog_matches:
        m = prog_matches[-1]
        latest_prog = {
            "epoch": int(m.group(1)),
            "percent": int(m.group(2)),
            "step": int(m.group(3)),
            "total": int(m.group(4)),
        }

    # Example: Epoch 1/100 (243.6s)
    p_summary = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\(([0-9.]+)s\)")
    summary_matches = list(p_summary.finditer(text))
    latest_summary = None
    if summary_matches:
        m = summary_matches[-1]
        latest_summary = {
            "epoch": int(m.group(1)),
            "total_epochs": int(m.group(2)),
            "epoch_time_sec": float(m.group(3)),
        }

    # Example: P2=pass P3=fail LR=1.00e-05
    p_gate = re.compile(r"P2=(pass|fail)\s+P3=(pass|fail)\s+LR=([0-9.eE+-]+)")
    gate_matches = list(p_gate.finditer(text))
    latest_gate = None
    if gate_matches:
        m = gate_matches[-1]
        latest_gate = {
            "p2": m.group(1),
            "p3": m.group(2),
            "lr": m.group(3),
        }

    warnings = {
        "audio_save_failed": "Audio save failed" in text,
        "torchcodec_error": "Could not load libtorchcodec" in text,
        "nan_detected": "NaN" in text or "nan" in text,
    }

    return {
        "exists": True,
        "latest_progress_bar": latest_prog,
        "latest_epoch_summary": latest_summary,
        "latest_gate_line": latest_gate,
        "warnings": warnings,
    }


def build_snapshot(run_dir: Path) -> Dict[str, Any]:
    cfg = load_json(run_dir / "config.json") or {}
    hist = load_json(run_dir / "metrics_history.json") or {}
    log = parse_log(run_dir / "train.log")

    epochs_logged = len(hist.get("train_total_loss", []))
    latest_metrics = None
    if epochs_logged > 0:
        i = epochs_logged - 1

        def safe_get(key: str):
            vals = hist.get(key, [])
            return vals[i] if i < len(vals) else None

        latest_metrics = {
            "epoch": epochs_logged,
            "train_total_loss": safe_get("train_total_loss"),
            "val_total_loss": safe_get("val_total_loss"),
            "feature_mse": safe_get("feature_mse"),
            "entropy": safe_get("entropy"),
            "top10_mass": safe_get("top10_mass"),
            "used_codes": safe_get("used_codes"),
            "p2_pass": safe_get("p2_pass"),
            "p3_pass": safe_get("p3_pass"),
        }

    return {
        "snapshot_time": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_exists": run_dir.exists(),
        "config_core": {
            "epochs": cfg.get("epochs"),
            "seed": cfg.get("seed"),
            "batch_size": cfg.get("batch_size"),
            "grad_accum": cfg.get("grad_accum"),
            "lora_rank": cfg.get("lora_rank"),
            "lora_alpha": cfg.get("lora_alpha"),
            "t453_min_weight": cfg.get("t453_min_weight"),
            "t453_ramp_epochs": cfg.get("t453_ramp_epochs"),
            "save_checkpoint_every": cfg.get("save_checkpoint_every"),
            "save_audio_interval": cfg.get("save_audio_interval"),
        },
        "epochs_logged": epochs_logged,
        "latest_logged_metrics": latest_metrics,
        "log_parse": log,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    snap = build_snapshot(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {args.out}")
    print(f"epochs_logged={snap['epochs_logged']}")
    lp = snap["log_parse"]["latest_progress_bar"]
    if lp:
        print(
            "latest_progress="
            f"epoch{lp['epoch']} {lp['step']}/{lp['total']} ({lp['percent']}%)"
        )
    lm = snap["latest_logged_metrics"]
    if lm:
        print(
            "latest_logged_metrics="
            f"epoch{lm['epoch']} mse={lm['feature_mse']:.6f} "
            f"P2={lm['p2_pass']} P3={lm['p3_pass']}"
        )


if __name__ == "__main__":
    main()

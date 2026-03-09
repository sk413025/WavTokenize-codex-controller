from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _latest_match(pattern: str, out_dir: Path) -> Path | None:
    matches = sorted(out_dir.glob(pattern))
    return matches[-1] if matches else None


def _hypothesis_transition(overall_state: str) -> Tuple[str, str]:
    if overall_state == "completed":
        return "run_completed", "default"
    if overall_state == "interrupted_but_usable":
        return "run_interrupted_but_usable", "default"
    if overall_state == "failed":
        return "run_needs_decision", "default"
    return "run_in_progress", "default"


def has_usable_artifacts(out_dir: Path) -> bool:
    history_path = out_dir / "history.json"
    history_has_val = False
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            history_has_val = bool(history.get("val_wav_mse"))
        except Exception:
            history_has_val = False
    return (
        history_has_val
        or (out_dir / "best_model.pt").exists()
        or _latest_match("checkpoint_epoch*.pt", out_dir) is not None
    )


def write_hypothesis_monitor_report(
    out_dir: Path,
    config: Dict[str, Any],
    *,
    overall_state: str,
    summary: str,
    completed_epochs: int,
    target_epochs: int,
    last_completed_epoch: int,
    best_val_mse: float,
    last_train_metrics: Dict[str, float] | None,
    last_val_metrics: Dict[str, float] | None,
    run_started_at: str,
    termination_reason: str | None = None,
) -> None:
    transition_class, next_owner = _hypothesis_transition(overall_state)
    latest_ckpt = _latest_match("checkpoint_epoch*.pt", out_dir)
    latest_plot = _latest_match("curves_epoch*.png", out_dir)
    best_model_path = out_dir / "best_model.pt"
    payload = {
        "status": "generated",
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(out_dir.resolve()),
        "run_started_at": run_started_at,
        "experiment": config["experiment"],
        "plan": config["plan"],
        "mode": config["mode"],
        "device": config.get("device"),
        "declared_next_action": config.get("launch_next_action"),
        "overall_state": overall_state,
        "transition_class": transition_class,
        "next_owner": next_owner,
        "summary": summary,
        "termination_reason": termination_reason,
        "completed_epochs": completed_epochs,
        "target_epochs": target_epochs,
        "last_completed_epoch": last_completed_epoch,
        "history_path": str((out_dir / "history.json").resolve()),
        "log_path": str((out_dir / "train.log").resolve()),
        "artifacts": {
            "config_json": str((out_dir / "config.json").resolve()),
            "history_json_exists": (out_dir / "history.json").exists(),
            "best_model_path": str(best_model_path.resolve()) if best_model_path.exists() else None,
            "latest_checkpoint_path": str(latest_ckpt.resolve()) if latest_ckpt else None,
            "latest_plot_path": str(latest_plot.resolve()) if latest_plot else None,
        },
        "latest_metrics": {
            "train": last_train_metrics,
            "val": last_val_metrics,
            "best_val_wav_mse": None if math.isinf(best_val_mse) else best_val_mse,
        },
    }
    _write_json(out_dir / "monitor_report.json", payload)


def write_hypothesis_analysis(
    out_dir: Path,
    config: Dict[str, Any],
    *,
    result_classification: str,
    summary: str,
    completed_epochs: int,
    target_epochs: int,
    best_val_mse: float,
    last_val_metrics: Dict[str, float] | None,
    termination_reason: str | None,
) -> None:
    next_action = "use_run_diagnosis"
    if result_classification == "clean_success":
        next_action = config.get("launch_next_action") or "review_results_in_codex_session"
    elif result_classification == "interrupted_but_usable":
        next_action = "review_partial_results_in_codex_session"

    baseline_val = config["baseline_val_wav_mse_best"]
    best_value = None if math.isinf(best_val_mse) else best_val_mse
    baseline_delta = None if best_value is None else best_value - baseline_val
    noisy_val = last_val_metrics.get("val_noisy_mse") if last_val_metrics else None
    latest_val = last_val_metrics.get("val_wav_mse") if last_val_metrics else None
    payload = {
        "status": "generated",
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(out_dir.resolve()),
        "experiment_id": config["experiment"],
        "plan": config["plan"],
        "result_classification": result_classification,
        "summary": summary,
        "completed_epochs": completed_epochs,
        "target_epochs": target_epochs,
        "best_val_wav_mse": best_value,
        "latest_val_wav_mse": latest_val,
        "termination_reason": termination_reason,
        "baseline_comparison": {
            "baseline_exp": config["baseline_exp"],
            "baseline_val_wav_mse_best": baseline_val,
            "delta_vs_baseline": baseline_delta,
            "beats_baseline": bool(best_value is not None and best_value < baseline_val),
        },
        "improves_over_noisy": None if latest_val is None or noisy_val is None else latest_val < noisy_val,
        "declared_next_action": config.get("launch_next_action"),
        "next_action": next_action,
    }
    _write_json(out_dir / "analysis.json", payload)

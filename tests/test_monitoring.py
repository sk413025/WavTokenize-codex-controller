from __future__ import annotations

import json
import time
from pathlib import Path

from codex_controller.monitor import inspect_run, write_monitor_result
from codex_controller.runtime import run_manifest


def _prepare_dry_run(patch_runtime_manifest, run_id: str) -> Path:
    manifest_path, _ = patch_runtime_manifest("experiments/manifests/exp0304_material_generalization_preflight.json")
    return run_manifest(manifest_path, run_id=run_id, dry_run=True)


def test_monitor_run_reports_completed_stage(patch_runtime_manifest, read_json) -> None:
    run_dir = _prepare_dry_run(patch_runtime_manifest, "monitor_completed")

    state = read_json(run_dir / "state.json")
    stage = state["stages"][0]
    stage["status"] = "completed"
    (run_dir / "preflight_report.json").write_text("{}\n", encoding="utf-8")
    Path(stage["log_path"]).write_text("PREFLIGHT STATUS: ready_for_smoke\n", encoding="utf-8")
    (run_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = inspect_run(run_dir, stall_seconds=1)
    assert report["overall_state"] == "completed"
    assert report["artifact_readiness"][stage["name"]]["completion_ready"] is True


def test_monitor_run_marks_stalled_stage(patch_runtime_manifest, read_json) -> None:
    run_dir = _prepare_dry_run(patch_runtime_manifest, "monitor_stalled")

    state = read_json(run_dir / "state.json")
    stage = state["stages"][0]
    stage["status"] = "running"
    log_path = Path(stage["log_path"])
    log_path.write_text("still running\n", encoding="utf-8")
    old = time.time() - 5
    __import__("os").utime(log_path, (old, old))
    (run_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_path = write_monitor_result(run_dir, stall_seconds=1)
    result = read_json(result_path)
    assert result["overall_state"] == "stalled"
    assert result["recommended_next_action"] == "diagnose_stalled_run"

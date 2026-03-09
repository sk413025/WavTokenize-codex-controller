from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .adapters import resolve_stage


class MonitorError(ValueError):
    """Raised when a monitor report cannot be produced."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inspect_run(run_dir_arg: str | Path, *, stall_seconds: int = 1800) -> Dict[str, Any]:
    run_dir = Path(run_dir_arg).resolve()
    state_path = run_dir / "state.json"
    manifest_path = run_dir / "manifest.snapshot.json"
    if not state_path.exists() or not manifest_path.exists():
        raise MonitorError(f"Run directory is missing controller state: {run_dir}")

    state = load_json(state_path)
    manifest = load_json(manifest_path)
    repo_root = Path(state["workspace_root"]).resolve()

    stage_reports: List[Dict[str, Any]] = []
    issues: List[str] = []
    readiness: Dict[str, Any] = {}
    selected = [stage for stage in state.get("stages", []) if stage.get("status") != "skipped"]

    for stage_state in selected:
        stage_def = resolve_stage(_stage_definition(manifest, stage_state["name"]), repo_root)
        report = inspect_stage(run_dir, repo_root, stage_state, stage_def, stall_seconds=stall_seconds)
        stage_reports.append(report)
        readiness[stage_state["name"]] = {
            "completion_ready": report["completion_ready"],
            "artifact_ready": report["artifact_ready"],
            "checkpoint_ready": report["checkpoint_ready"],
            "output_dir": report.get("output_dir"),
        }
        if report.get("failure_signature"):
            issues.append(f"{stage_state['name']}:{report['failure_signature']}")

    overall_state = classify_run_state(stage_reports)
    if overall_state == "completed":
        recommendation = "handoff_to_analyst"
        summary = "All selected stages have completion evidence and expected artifacts."
    elif overall_state == "failed":
        recommendation = "diagnose_execution_failure"
        summary = "At least one stage failed or is missing required completion evidence."
    elif overall_state == "stalled":
        recommendation = "diagnose_stalled_run"
        summary = "At least one running stage appears stalled or stopped updating its log."
    elif overall_state == "running":
        recommendation = "continue_monitoring"
        summary = "At least one stage is still running and making observable progress."
    else:
        recommendation = "wait_for_stage_start"
        summary = "No selected stage has started yet."

    transition_class, next_owner = classify_transition(overall_state)

    return {
        "status": "generated",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "overall_state": overall_state,
        "transition_class": transition_class,
        "next_owner": next_owner,
        "summary": summary,
        "recommended_next_action": recommendation,
        "issues": issues,
        "stage_reports": stage_reports,
        "artifact_readiness": readiness,
    }


def inspect_stage(
    run_dir: Path,
    repo_root: Path,
    stage_state: Dict[str, Any],
    stage_def: Dict[str, Any],
    *,
    stall_seconds: int,
) -> Dict[str, Any]:
    log_path = Path(stage_state["log_path"])
    log_exists = log_path.exists()
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_exists else ""
    marker_hits = [marker for marker in (stage_def.get("completion") or {}).get("stdout_contains", []) if marker in log_text]
    completion_files = rendered_completion_files(stage_def, repo_root, run_dir)
    completion_missing = [str(path) for path in completion_files if not path.exists()]
    output_dir = rendered_output_dir(stage_def, repo_root, run_dir)
    expected_output_files = []
    output_missing = []
    checkpoint_ready = False
    if output_dir is not None:
        for relative in stage_def.get("artifacts", {}).get("expected_files", []):
            target = (output_dir / relative).resolve()
            expected_output_files.append(str(target))
            if not target.exists():
                output_missing.append(str(target))
        checkpoint_ready = any(output_dir.glob("checkpoint_epoch*.pt")) or (output_dir / "best_model.pt").exists() or (output_dir / "best_model_val_total.pt").exists()

    log_age_seconds = None
    if log_exists:
        log_age_seconds = max(0.0, datetime.now().timestamp() - log_path.stat().st_mtime)

    execution_state = "planned"
    failure_signature = None
    status = stage_state.get("status")
    if status == "completed":
        if completion_missing:
            execution_state = "failed"
            failure_signature = "missing_completion_file"
        elif (stage_def.get("completion") or {}).get("stdout_contains") and not marker_hits:
            execution_state = "failed"
            failure_signature = "missing_completion_marker"
        elif output_missing and expected_output_files:
            execution_state = "failed"
            failure_signature = "missing_output_artifact"
        else:
            execution_state = "completed"
    elif status == "failed":
        execution_state = "failed"
        failure_signature = "stage_failed"
    elif status == "running":
        if not log_exists:
            execution_state = "running"
            failure_signature = "log_missing"
        elif log_age_seconds is not None and log_age_seconds > stall_seconds:
            execution_state = "stalled"
            failure_signature = "stalled_log"
        else:
            execution_state = "running"
    elif status == "planned":
        execution_state = "planned"
    else:
        execution_state = status or "unknown"

    return {
        "stage": stage_state["name"],
        "status": status,
        "execution_state": execution_state,
        "log_path": str(log_path),
        "log_exists": log_exists,
        "log_age_seconds": log_age_seconds,
        "completion_markers_seen": marker_hits,
        "completion_ready": not completion_missing and (bool(marker_hits) or not (stage_def.get("completion") or {}).get("stdout_contains")),
        "completion_missing": completion_missing,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "expected_output_files": expected_output_files,
        "missing_output_files": output_missing,
        "artifact_ready": not output_missing,
        "checkpoint_ready": checkpoint_ready,
        "failure_signature": failure_signature,
    }


def rendered_completion_files(stage_def: Dict[str, Any], repo_root: Path, run_dir: Path) -> List[Path]:
    completion = stage_def.get("completion") or {}
    rendered: List[Path] = []
    for value in completion.get("file_exists", []):
        path = Path(str(value).format(repo_root=str(repo_root), run_dir=str(run_dir)))
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        rendered.append(path)
    return rendered


def rendered_output_dir(stage_def: Dict[str, Any], repo_root: Path, run_dir: Path) -> Path | None:
    args = [render_arg(arg, repo_root, run_dir) for arg in stage_def.get("args", [])]
    for idx, value in enumerate(args[:-1]):
        if value == "--output_dir":
            path = Path(args[idx + 1])
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            return path
    command = stage_def.get("command")
    if command:
        parts = shlex.split(str(command).format(repo_root=str(repo_root), run_dir=str(run_dir)))
        for idx, value in enumerate(parts[:-1]):
            if value == "--output_dir":
                path = Path(parts[idx + 1])
                if not path.is_absolute():
                    path = (repo_root / path).resolve()
                return path
    return None


def classify_run_state(stage_reports: List[Dict[str, Any]]) -> str:
    if any(report["execution_state"] == "failed" for report in stage_reports):
        return "failed"
    if any(report["execution_state"] == "stalled" for report in stage_reports):
        return "stalled"
    if stage_reports and all(report["execution_state"] == "completed" for report in stage_reports):
        return "completed"
    if any(report["execution_state"] == "running" for report in stage_reports):
        return "running"
    return "planned"


def classify_transition(overall_state: str) -> tuple[str, str]:
    if overall_state == "completed":
        return "run_completed", "analyst"
    if overall_state in {"failed", "stalled"}:
        return "run_needs_decision", "default"
    return "run_in_progress", "default"


def write_monitor_result(
    run_dir_arg: str | Path,
    *,
    result_path_arg: str | Path | None = None,
    stall_seconds: int = 1800,
) -> Path:
    run_dir = Path(run_dir_arg).resolve()
    report = inspect_run(run_dir, stall_seconds=stall_seconds)
    result_path = Path(result_path_arg).resolve() if result_path_arg else (run_dir / "monitor_report.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_path


def render_arg(value: Any, repo_root: Path, run_dir: Path) -> str:
    rendered = str(value).format(repo_root=str(repo_root), run_dir=str(run_dir))
    if rendered.startswith("./"):
        return str((repo_root / rendered[2:]).resolve())
    return rendered


def _stage_definition(manifest: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    for stage in manifest.get("stages", []):
        if stage.get("name") == stage_name:
            return stage
    raise MonitorError(f"Unknown stage in manifest snapshot: {stage_name}")

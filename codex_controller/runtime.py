from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .adapters import resolve_stage
from .knowledge import load_knowledge_context, update_knowledge
from .manifest import load_manifest
from .monitor import inspect_run


class RuntimeErrorWithContext(RuntimeError):
    """Raised for run-time controller errors."""


STAGE_STATUSES = {"planned", "running", "completed", "failed", "skipped"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(experiment_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = experiment_id.replace("/", "_")
    return f"{safe}_{stamp}"


def run_manifest(
    manifest_path_arg: str | Path,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
    from_stage: str | None = None,
    through_stage: str | None = None,
) -> Path:
    manifest_path, manifest, repo_root = load_manifest(manifest_path_arg)
    run_dir, state = _create_run_context(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=repo_root,
        run_id=run_id,
        from_stage=from_stage,
        through_stage=through_stage,
        dry_run=dry_run,
    )

    _initialize_run_artifacts(run_dir, manifest)
    _persist_state(run_dir, state)
    _append_index(run_dir.parent, _index_record(state))
    _append_event(run_dir, {"event": "run_created", "run_id": state["run_id"], "dry_run": dry_run})

    if dry_run:
        state["status"] = "planned"
        state["run_status_detail"] = "prepared"
        state["result_classification"] = "dry_run"
        state["next_action"] = "review_manifest_in_codex_session"
        _write_json(run_dir / "analysis.json", _dry_run_analysis(state))
        _persist_state(run_dir, state)
        _append_index(run_dir.parent, _index_record(state))
        return run_dir

    execution_error: Exception | None = None
    try:
        state["status"] = "running"
        state["run_status_detail"] = "executing"
        _persist_state(run_dir, state)
        _append_event(run_dir, {"event": "run_started", "run_id": state["run_id"]})
        for stage in state["stages"]:
            if stage["status"] == "skipped":
                continue
            _ensure_dependencies_satisfied(stage, state)
            _run_stage(stage, manifest, repo_root, run_dir, state)
            _persist_state(run_dir, state)
    except Exception as exc:  # noqa: BLE001
        execution_error = exc
        state["status"] = "failed"
        state["failure_reason"] = str(exc)
        state["run_status_detail"] = "failed"
        _append_event(run_dir, {"event": "run_failed", "error": str(exc)})
        _persist_state(run_dir, state)

    _finalize_run_artifacts(
        run_dir=run_dir,
        state=state,
        manifest=manifest,
        repo_root=repo_root,
        execution_error=execution_error,
    )

    if execution_error is not None:
        raise RuntimeErrorWithContext(str(execution_error))
    return run_dir


def resume_run(run_dir_arg: str | Path, *, dry_run: bool = False, through_stage: str | None = None) -> Path:
    run_dir = Path(run_dir_arg).resolve()
    state_path = run_dir / "state.json"
    manifest_snapshot = run_dir / "manifest.snapshot.json"
    if not state_path.exists() or not manifest_snapshot.exists():
        raise RuntimeErrorWithContext(f"Run directory is missing controller state: {run_dir}")

    state = _read_json(state_path)
    manifest = _read_json(manifest_snapshot)
    repo_root = Path(state["workspace_root"]).resolve()
    selected = _select_stage_names(
        manifest["stages"],
        from_stage=_first_resume_stage_name(state),
        through_stage=through_stage,
    )
    if not selected:
        raise RuntimeErrorWithContext(f"No resumable stages found for {run_dir}")

    _append_event(run_dir, {"event": "resume_requested", "selected_stages": selected, "dry_run": dry_run})
    if dry_run:
        return run_dir

    state["status"] = "running"
    state["run_status_detail"] = "resumed"
    _persist_state(run_dir, state)

    execution_error: Exception | None = None
    try:
        for stage in state["stages"]:
            if stage["name"] not in selected:
                continue
            _ensure_dependencies_satisfied(stage, state)
            _run_stage(stage, manifest, repo_root, run_dir, state)
            _persist_state(run_dir, state)
    except Exception as exc:  # noqa: BLE001
        execution_error = exc
        state["status"] = "failed"
        state["failure_reason"] = str(exc)
        state["run_status_detail"] = "failed"
        _persist_state(run_dir, state)

    _finalize_run_artifacts(
        run_dir=run_dir,
        state=state,
        manifest=manifest,
        repo_root=repo_root,
        execution_error=execution_error,
    )
    if execution_error is not None:
        raise RuntimeErrorWithContext(str(execution_error))
    return run_dir


def summarize_run(run_dir_arg: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir_arg).resolve()
    state = _read_json(run_dir / "state.json")
    return {
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "status": state["status"],
        "run_status_detail": _run_status_detail(state),
        "result_classification": state.get("result_classification"),
        "next_action": state.get("next_action"),
        "failure_reason": state.get("failure_reason"),
        "stages": _stage_status_map(state),
        "metrics": _read_optional_json(run_dir / "metrics.json"),
        "analysis": _read_optional_json(run_dir / "analysis.json"),
        "diagnosis": _read_optional_json(run_dir / "diagnosis.json"),
        "monitor_report": _read_optional_json(run_dir / "monitor_report.json"),
    }


def _create_run_context(
    *,
    manifest: Dict[str, Any],
    manifest_path: Path,
    repo_root: Path,
    run_id: str | None,
    from_stage: str | None,
    through_stage: str | None,
    dry_run: bool,
) -> tuple[Path, Dict[str, Any]]:
    run_root = (repo_root / manifest.get("run_root", "controller_runs")).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or generate_run_id(manifest["experiment_id"])
    run_dir = run_root / resolved_run_id
    if run_dir.exists():
        raise RuntimeErrorWithContext(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    selected = _select_stage_names(manifest["stages"], from_stage=from_stage, through_stage=through_stage)
    state = _initial_state(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=repo_root,
        run_dir=run_dir,
        run_id=resolved_run_id,
        selected=selected,
        dry_run=dry_run,
    )
    return run_dir, state


def _initial_state(
    *,
    manifest: Dict[str, Any],
    manifest_path: Path,
    repo_root: Path,
    run_dir: Path,
    run_id: str,
    selected: List[str],
    dry_run: bool,
) -> Dict[str, Any]:
    stages = []
    for stage in manifest["stages"]:
        resolved = resolve_stage(stage, repo_root)
        stages.append(
            {
                "name": stage["name"],
                "adapter": resolved["adapter"],
                "adapter_id": stage.get("adapter_id"),
                "after": stage.get("after", []),
                "status": "planned" if stage["name"] in selected else "skipped",
                "description": stage.get("description", ""),
                "command": _command_preview(resolved, repo_root, run_dir),
                "log_path": str((run_dir / f"{stage['name']}.log").resolve()),
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "artifacts": resolved.get("artifacts", {}),
                "known_failures": resolved.get("known_failures", []),
            }
        )

    return {
        "run_id": run_id,
        "experiment_id": manifest["experiment_id"],
        "manifest_path": str(manifest_path),
        "workspace_root": str(repo_root),
        "run_dir": str(run_dir),
        "status": "planned",
        "run_status_detail": "planning",
        "dry_run": dry_run,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "failure_reason": None,
        "hypothesis": manifest["hypothesis"],
        "baseline_refs": manifest.get("baseline_refs", []),
        "result_classification": "planned",
        "next_action": None,
        "supersedes": manifest.get("supersedes"),
        "stages": stages,
    }


def _initialize_run_artifacts(run_dir: Path, manifest: Dict[str, Any]) -> None:
    _write_json(run_dir / "manifest.snapshot.json", manifest)
    _write_json(run_dir / "metrics.json", _default_placeholder("metrics"))
    _write_json(run_dir / "analysis.json", _default_placeholder("analysis"))
    _write_json(run_dir / "diagnosis.json", _default_placeholder("diagnosis"))
    _write_json(run_dir / "monitor_report.json", _default_placeholder("monitor_report"))
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")


def _finalize_run_artifacts(
    *,
    run_dir: Path,
    state: Dict[str, Any],
    manifest: Dict[str, Any],
    repo_root: Path,
    execution_error: Exception | None,
) -> None:
    monitor_report = inspect_run(run_dir)
    _write_json(run_dir / "monitor_report.json", monitor_report)

    metrics = _build_metrics_snapshot(run_dir=run_dir, state=state, manifest=manifest, monitor_report=monitor_report)
    _write_json(run_dir / "metrics.json", metrics)

    analysis = _build_analysis(
        state=state,
        manifest=manifest,
        run_dir=run_dir,
        execution_error=execution_error,
        monitor_report=monitor_report,
    )
    _write_json(run_dir / "analysis.json", analysis)
    state["result_classification"] = analysis["result_classification"]
    state["next_action"] = analysis["next_action"]

    diagnosis = _default_placeholder("diagnosis")
    if _should_diagnose(analysis):
        diagnosis = _build_diagnosis(state)
    _write_json(run_dir / "diagnosis.json", diagnosis)

    if execution_error is not None:
        state["status"] = "failed"
    else:
        state["status"] = "completed"
    state["finished_at"] = utc_now()
    state["run_status_detail"] = "completed"

    update_knowledge(repo_root, state, analysis, manifest)
    _persist_state(run_dir, state)
    _append_index(run_dir.parent, _index_record(state))


def _build_metrics_snapshot(
    *,
    run_dir: Path,
    state: Dict[str, Any],
    manifest: Dict[str, Any],
    monitor_report: Dict[str, Any],
) -> Dict[str, Any]:
    external = _read_optional_json(run_dir / "metrics.json") or {}
    if external.get("status") == "not_generated":
        external = {}

    observed = external.get("observed_metrics") if isinstance(external.get("observed_metrics"), dict) else external
    readiness = monitor_report.get("artifact_readiness", {})
    return {
        "schema_version": 1,
        "status": "generated",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "family": manifest["family"],
        "tier": _run_tier(manifest["experiment_id"]),
        "external_metrics_present": bool(external),
        "monitor_overall_state": monitor_report.get("overall_state"),
        "artifacts_ready": all(item.get("artifact_ready", False) for item in readiness.values()) if readiness else False,
        "observed_metrics": observed if isinstance(observed, dict) else {},
    }


def _dry_run_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "generated",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "result_classification": "dry_run",
        "summary": "Dry-run completed. Review the manifest, commands, and stage dependencies in the Codex session before launching.",
        "unmet_reasons": [],
        "suggested_changes": [],
        "next_action": "review_manifest_in_codex_session",
    }


def _build_analysis(
    *,
    state: Dict[str, Any],
    manifest: Dict[str, Any],
    run_dir: Path,
    execution_error: Exception | None,
    monitor_report: Dict[str, Any],
) -> Dict[str, Any]:
    acceptance = manifest.get("acceptance_criteria", {})
    unmet_reasons: List[str] = []
    required_stage_status = acceptance.get("required_stage_status", {})
    stage_statuses = _stage_status_map(state)
    for stage_name, expected in required_stage_status.items():
        if stage_statuses.get(stage_name) != expected:
            unmet_reasons.append(f"stage:{stage_name} expected {expected} got {stage_statuses.get(stage_name)}")

    for required in acceptance.get("required_files", []):
        target = Path(state["workspace_root"]) / required
        if not target.exists():
            unmet_reasons.append(f"missing_file:{required}")

    monitor_state = monitor_report.get("overall_state")
    if monitor_state in {"failed", "stalled"}:
        unmet_reasons.append(f"monitor_state:{monitor_state}")

    if execution_error is not None or monitor_state in {"failed", "stalled"}:
        classification = "failed"
        summary = f"Run failed during execution: {state.get('failure_reason') or execution_error}"
        next_action = "use_run_diagnosis"
    elif unmet_reasons:
        classification = "needs_iteration"
        summary = "Run completed execution but did not satisfy all acceptance markers. Review logs and artifacts before the next change."
        next_action = "review_with_run_diagnosis_and_result_comparison"
    else:
        classification = acceptance.get("result_classification_on_pass", "candidate")
        summary = "Run satisfied the manifest execution contract. Review logs, metrics, and artifacts in the Codex session."
        next_action = "review_results_in_codex_session"

    return {
        "status": "generated",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "result_classification": classification,
        "summary": summary,
        "metrics_ref": str((run_dir / "metrics.json").resolve()),
        "monitor_report_ref": str((run_dir / "monitor_report.json").resolve()),
        "unmet_reasons": unmet_reasons,
        "suggested_changes": _suggested_changes_from_failures(state),
        "next_action": next_action,
    }


def _build_diagnosis(state: Dict[str, Any]) -> Dict[str, Any]:
    suggestions = _suggested_changes_from_failures(state)
    summary = state.get("failure_reason") or "Acceptance markers were unmet without a hard execution failure."
    return {
        "status": "generated",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "summary": summary,
        "suspected_causes": suggestions or ["Inspect stage logs and compare against known failure signatures."],
        "advisory": True,
    }


def _should_diagnose(analysis: Dict[str, Any]) -> bool:
    return analysis["result_classification"] in {"failed", "needs_iteration"}


def _run_tier(experiment_id: str) -> str:
    if "preflight" in experiment_id:
        return "preflight"
    if "smoke" in experiment_id:
        return "smoke"
    if "short" in experiment_id:
        return "short"
    return "full"


def _suggested_changes_from_failures(state: Dict[str, Any]) -> List[str]:
    suggestions: List[str] = []
    for stage in state["stages"]:
        if stage["status"] != "failed":
            continue
        for failure in stage.get("known_failures", []):
            suggestions.extend(failure.get("suggested_changes", []))
    return list(dict.fromkeys(suggestions))


def _run_stage(stage_state: Dict[str, Any], manifest: Dict[str, Any], repo_root: Path, run_dir: Path, state: Dict[str, Any]) -> None:
    stage_def = resolve_stage(_stage_definition(manifest, stage_state["name"]), repo_root)
    command = _build_command(stage_def, repo_root, run_dir)
    env = _build_env(manifest, stage_def, repo_root, run_dir)
    cwd = _resolve_cwd(stage_def, repo_root, run_dir)
    log_path = run_dir / f"{stage_state['name']}.log"

    stage_state["status"] = "running"
    stage_state["started_at"] = utc_now()
    _append_event(run_dir, {"event": "stage_started", "stage": stage_state["name"], "command": command, "cwd": str(cwd)})

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"# stage={stage_state['name']}\n")
        log_handle.write(f"# cwd={cwd}\n")
        log_handle.write(f"# command={' '.join(shlex.quote(part) for part in command)}\n")
        log_handle.write(f"# started_at={stage_state['started_at']}\n\n")
        completed = subprocess.run(command, cwd=str(cwd), env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)

    stage_state["finished_at"] = utc_now()
    stage_state["returncode"] = completed.returncode
    if completed.returncode != 0:
        stage_state["status"] = "failed"
        _append_event(run_dir, {"event": "stage_failed", "stage": stage_state['name'], "returncode": completed.returncode, "log_path": str(log_path)})
        raise RuntimeErrorWithContext(f"Stage {stage_state['name']} failed with return code {completed.returncode}. See {log_path}")

    _check_completion(stage_def, repo_root, run_dir, log_path, stage_state["name"])
    stage_state["status"] = "completed"
    _append_event(run_dir, {"event": "stage_completed", "stage": stage_state['name'], "log_path": str(log_path)})


def _check_completion(stage: Dict[str, Any], repo_root: Path, run_dir: Path, log_path: Path, stage_name: str) -> None:
    completion = stage.get("completion") or {}
    required_files = completion.get("file_exists", [])
    for relative_path in required_files:
        rendered = str(relative_path).format(repo_root=str(repo_root), run_dir=str(run_dir))
        target = Path(rendered)
        if not target.is_absolute():
            target = (repo_root / rendered).resolve()
        if not target.exists():
            raise RuntimeErrorWithContext(f"Stage {stage_name} finished but required artifact is missing: {target}")

    stdout_contains = completion.get("stdout_contains", [])
    if stdout_contains:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if not any(marker in text for marker in stdout_contains):
            raise RuntimeErrorWithContext(f"Stage {stage_name} finished but log {log_path} is missing completion marker {stdout_contains}")


def _select_stage_names(stages: List[Dict[str, Any]], *, from_stage: str | None, through_stage: str | None) -> List[str]:
    names = [stage["name"] for stage in stages]
    start = 0
    end = len(names) - 1
    if from_stage is not None:
        if from_stage not in names:
            raise RuntimeErrorWithContext(f"Unknown from-stage: {from_stage}")
        start = names.index(from_stage)
    if through_stage is not None:
        if through_stage not in names:
            raise RuntimeErrorWithContext(f"Unknown through-stage: {through_stage}")
        end = names.index(through_stage)
    if start > end:
        raise RuntimeErrorWithContext("from-stage occurs after through-stage")
    return names[start : end + 1]


def _ensure_dependencies_satisfied(stage_state: Dict[str, Any], state: Dict[str, Any]) -> None:
    status_by_name = {stage["name"]: stage["status"] for stage in state["stages"]}
    for dependency in stage_state.get("after", []):
        if status_by_name.get(dependency) != "completed":
            raise RuntimeErrorWithContext(f"Stage {stage_state['name']} cannot run before dependency {dependency} is completed")


def _build_command(stage: Dict[str, Any], repo_root: Path, run_dir: Path | None = None) -> List[str]:
    adapter = stage["adapter"]
    if adapter == "python_script":
        python_exe = stage.get("python") or sys.executable
        entrypoint = (repo_root / stage["entrypoint"]).resolve()
        args = [_render_arg(arg, repo_root, run_dir) for arg in stage.get("args", [])]
        return [python_exe, str(entrypoint), *args]
    if adapter == "shell":
        return ["bash", "-lc", _render_arg(stage["command"], repo_root, run_dir)]
    raise RuntimeErrorWithContext(f"Unsupported adapter during execution: {adapter}")


def _command_preview(stage: Dict[str, Any], repo_root: Path, run_dir: Path | None = None) -> List[str]:
    return _build_command(stage, repo_root, run_dir)


def _build_env(manifest: Dict[str, Any], stage: Dict[str, Any], repo_root: Path, run_dir: Path | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    merged = {}
    merged.update(manifest.get("default_env", {}))
    merged.update(stage.get("env", {}))
    for key, value in merged.items():
        if key == "PYTHONPATH":
            previous = env.get("PYTHONPATH", "")
            rendered = value.format(repo_root=str(repo_root), run_root=manifest.get("run_root", "controller_runs"), run_dir=str(run_dir) if run_dir else "")
            resolved = str((repo_root / rendered).resolve()) if rendered and not Path(rendered).is_absolute() else (rendered or str(repo_root))
            env[key] = resolved if not previous else f"{resolved}:{previous}"
        else:
            env[key] = value.format(repo_root=str(repo_root), run_root=manifest.get("run_root", "controller_runs"), run_dir=str(run_dir) if run_dir else "")
    env.setdefault("WAVTOKENIZE_REPO_ROOT", str(repo_root))
    if run_dir is not None:
        env.setdefault("WAVTOKENIZE_RUN_DIR", str(run_dir))
    return env


def _resolve_cwd(stage: Dict[str, Any], repo_root: Path, run_dir: Path | None = None) -> Path:
    cwd = stage.get("cwd")
    if not cwd:
        return repo_root
    rendered = str(cwd).format(repo_root=str(repo_root), run_dir=str(run_dir) if run_dir else "")
    path = Path(rendered)
    return path.resolve() if path.is_absolute() else (repo_root / rendered).resolve()


def _render_arg(value: Any, repo_root: Path, run_dir: Path | None) -> str:
    if isinstance(value, str):
        return value.format(repo_root=str(repo_root), run_dir=str(run_dir) if run_dir else "")
    return str(value)


def _stage_definition(manifest: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    for stage in manifest["stages"]:
        if stage["name"] == stage_name:
            return stage
    raise RuntimeErrorWithContext(f"Stage definition not found for {stage_name}")


def _first_resume_stage_name(state: Dict[str, Any]) -> str | None:
    for stage in state["stages"]:
        if stage["status"] in {"planned", "failed"}:
            return stage["name"]
    return None


def _stage_status_map(state: Dict[str, Any]) -> Dict[str, str]:
    return {stage["name"]: stage["status"] for stage in state["stages"]}


def _run_status_detail(state: Dict[str, Any]) -> Any:
    return state.get("run_status_detail", state.get("controller_status"))


def _append_index(run_root: Path, record: Dict[str, Any]) -> None:
    index_path = run_root / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _index_record(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ts": utc_now(),
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "run_dir": state["run_dir"],
        "status": state["status"],
        "run_status_detail": _run_status_detail(state),
        "result_classification": state.get("result_classification"),
        "manifest_path": state["manifest_path"],
    }


def _append_event(run_dir: Path, event: Dict[str, Any]) -> None:
    payload = {"ts": utc_now(), **event}
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _persist_state(run_dir: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    _write_json(run_dir / "state.json", state)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _default_placeholder(kind: str) -> Dict[str, Any]:
    return {
        "status": "not_generated",
        "kind": kind,
        "generated_at": None,
        "summary": None,
    }

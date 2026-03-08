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
from .agents import build_controller_graph, bootstrap_agent_state, load_agent_registry, set_active_agent, update_graph_node
from .dispatch import (
    build_dispatch_plan,
    classify_result_status,
    enrich_controller_graph,
    graph_node_by_name,
    packet_payload,
)
from .knowledge import load_knowledge_context, update_knowledge
from .manifest import load_manifest

RUN_STATUSES = {"planned", "running", "failed", "completed", "diagnosed", "superseded"}
STAGE_STATUSES = {"planned", "running", "completed", "failed", "skipped"}
DISPATCH_PENDING = {"ready", "delegated", "reported"}
NODE_ORDER = ["plan", "prepare", "execute", "monitor", "analyze", "diagnose", "patch", "propose_next", "queue_next"]


class RuntimeErrorWithContext(RuntimeError):
    """Raised for run-time controller errors."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(experiment_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = experiment_id.replace("/", "_")
    return f"{safe}_{stamp}"


def prepare_run(
    manifest_path_arg: str | Path,
    *,
    run_id: str | None = None,
    from_stage: str | None = None,
    through_stage: str | None = None,
) -> Path:
    manifest_path, manifest, repo_root = load_manifest(manifest_path_arg)
    registry = load_agent_registry(repo_root)
    knowledge_context = load_knowledge_context(repo_root, manifest["family"])
    run_dir, state = _create_run_context(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=repo_root,
        registry=registry,
        knowledge_context=knowledge_context,
        run_id=run_id,
        from_stage=from_stage,
        through_stage=through_stage,
        dry_run=False,
        dispatch_mode="native_multi_agent",
    )
    _initialize_run_artifacts(run_dir, manifest, registry, state)
    _mark_node_ready(state, "plan", reason="Run prepared; ready for planner packet emission.")
    state["controller_status"] = "dispatch_ready"
    _persist_state(run_dir, state)
    _append_index(run_dir.parent, _index_record(state))
    _append_event(run_dir, {"event": "run_prepared", "run_id": state["run_id"]})
    _append_decision(run_dir, "codex", "run_prepared", {"dispatch_mode": state["dispatch_mode"]})
    _append_action(run_dir, "codex", "prepare_run", {"run_id": state["run_id"]})
    _refresh_dispatch_views(run_dir, state, manifest, registry)
    return run_dir


def emit_packets(run_dir_arg: str | Path, *, nodes: List[str] | None = None) -> List[Path]:
    run_dir, state, manifest, registry, _ = _load_run_context(run_dir_arg)
    emitted: List[Path] = []
    for node in state["controller_graph"]:
        if nodes and node["node"] not in nodes:
            continue
        if node.get("dispatch_status") != "ready":
            continue
        packet = packet_payload(state=state, manifest=manifest, node=node, registry=registry)
        packet_path = Path(node["packet_path"])
        _write_json(packet_path, packet)
        node["dispatch_status"] = "delegated"
        node["delegated_at"] = utc_now()
        update_graph_node(state, node["node"], status="running", timestamp=utc_now(), note="Packet emitted to native role.")
        if node["packet_id"] not in state["active_packet_ids"]:
            state["active_packet_ids"].append(node["packet_id"])
        if node["agent"] not in state["awaiting_results_from"]:
            state["awaiting_results_from"].append(node["agent"])
        _append_action(run_dir, "codex", "emit_packet", {"node": node["node"], "packet_id": node["packet_id"]})
        emitted.append(packet_path)

    if emitted:
        state["controller_status"] = "dispatch_delegated"
        _persist_state(run_dir, state)
        _refresh_dispatch_views(run_dir, state, manifest, registry)
    return emitted


def ingest_agent_result(run_dir_arg: str | Path, *, node_name: str, result_path_arg: str | Path) -> Path:
    run_dir, state, manifest, registry, _ = _load_run_context(run_dir_arg)
    node = graph_node_by_name(state, node_name)
    result_path = Path(result_path_arg).resolve()
    if not result_path.exists():
        raise RuntimeErrorWithContext(f"Agent result does not exist: {result_path}")
    result = _read_json(result_path)
    target_path = Path(node["result_path"])
    _write_json(target_path, result)
    node["dispatch_status"] = "reported"
    node["reported_at"] = utc_now()
    if node["packet_id"] in state["active_packet_ids"]:
        state["active_packet_ids"].remove(node["packet_id"])
    if node["agent"] in state["awaiting_results_from"]:
        state["awaiting_results_from"].remove(node["agent"])
    if node["packet_id"] not in state["reported_results"]:
        state["reported_results"].append(node["packet_id"])
    state["controller_status"] = "dispatch_reported"
    _append_decision(run_dir, node["agent"], "agent_result_reported", {"node": node_name, "result_path": str(target_path)})
    _append_action(run_dir, "codex", "ingest_agent_result", {"node": node_name, "result_path": str(target_path)})
    _persist_state(run_dir, state)
    _refresh_dispatch_views(run_dir, state, manifest, registry)
    return target_path


def advance_run(run_dir_arg: str | Path) -> Path:
    run_dir, state, manifest, registry, knowledge_context = _load_run_context(run_dir_arg)
    progressed = False
    for node in state["controller_graph"]:
        if node.get("dispatch_status") != "reported":
            continue
        result = _read_json(Path(node["result_path"]))
        decision = classify_result_status(result)
        payload = result.get("structured_output", {})
        if decision == "rejected":
            node["dispatch_status"] = "rejected"
            if node["packet_id"] not in state["rejected_results"]:
                state["rejected_results"].append(node["packet_id"])
            state["next_action"] = f"reissue_packet:{node['node']}"
            _append_action(run_dir, "codex", "reject_result", {"node": node["node"], "packet_id": node["packet_id"]})
            progressed = True
            continue
        if decision == "superseded":
            node["dispatch_status"] = "superseded"
            if node["packet_id"] not in state["superseded_results"]:
                state["superseded_results"].append(node["packet_id"])
            state["next_action"] = f"superseded:{node['node']}"
            _append_action(run_dir, "codex", "supersede_result", {"node": node["node"], "packet_id": node["packet_id"]})
            progressed = True
            continue

        node["dispatch_status"] = "accepted"
        if node["packet_id"] not in state["accepted_results"]:
            state["accepted_results"].append(node["packet_id"])
        if node["packet_id"] in state["reported_results"]:
            state["reported_results"].remove(node["packet_id"])
        _append_action(run_dir, "codex", "accept_result", {"node": node["node"], "packet_id": node["packet_id"]})
        _apply_node_result(run_dir, state, manifest, knowledge_context, node, result, payload)
        progressed = True
        break

    if not progressed:
        raise RuntimeErrorWithContext(f"No reported packets available to advance in {run_dir}")

    _persist_state(run_dir, state)
    _refresh_dispatch_views(run_dir, state, manifest, registry)
    return run_dir


def finalize_run(run_dir_arg: str | Path) -> Path:
    run_dir, state, manifest, registry, _ = _load_run_context(run_dir_arg)
    pending = [node["node"] for node in state["controller_graph"] if node.get("dispatch_status") in DISPATCH_PENDING]
    if pending:
        raise RuntimeErrorWithContext(f"Run still has pending dispatch work: {pending}")

    analysis = _read_optional_json(run_dir / "analysis.json") or _default_placeholder("analysis")
    if analysis.get("status") == "generated" and not state.get("knowledge_updated"):
        repo_root = Path(state["workspace_root"])
        update_knowledge(repo_root, state, analysis, manifest)
        state["knowledge_updated"] = True

    if state.get("result_classification") in {"failed", "needs_iteration"}:
        state["status"] = "diagnosed" if _node_dispatch_status(state, "diagnose") == "accepted" else "failed"
    else:
        state["status"] = "completed"

    state["controller_status"] = "completed"
    state["finished_at"] = utc_now()
    _append_action(run_dir, "codex", "finalize_run", {"status": state["status"]})
    _persist_state(run_dir, state)
    _append_index(run_dir.parent, _index_record(state))
    _refresh_dispatch_views(run_dir, state, manifest, registry)
    return run_dir


def run_manifest(
    manifest_path_arg: str | Path,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
    from_stage: str | None = None,
    through_stage: str | None = None,
) -> Path:
    manifest_path, manifest, repo_root = load_manifest(manifest_path_arg)
    registry = load_agent_registry(repo_root)
    knowledge_context = load_knowledge_context(repo_root, manifest["family"])
    run_dir, state = _create_run_context(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=repo_root,
        registry=registry,
        knowledge_context=knowledge_context,
        run_id=run_id,
        from_stage=from_stage,
        through_stage=through_stage,
        dry_run=dry_run,
        dispatch_mode="compatibility_run",
    )

    _initialize_run_artifacts(run_dir, manifest, registry, state)
    _persist_state(run_dir, state)
    _append_index(run_dir.parent, _index_record(state))
    _append_event(run_dir, {"event": "run_created", "run_id": state["run_id"], "selected_stages": [stage["name"] for stage in state["stages"] if stage["status"] != "skipped"], "dry_run": dry_run})
    _append_decision(run_dir, "codex", "run_created", {"dispatch_mode": state["dispatch_mode"], "dry_run": dry_run})
    _append_action(run_dir, "codex", "compatibility_run_created", {"run_id": state["run_id"]})

    _advance_node(run_dir, state, node="plan", agent="planner", note="Manifest accepted for compatibility execution.")
    _append_decision(run_dir, "planner", "manifest_loaded", {"hypothesis": manifest["hypothesis"]})
    _finish_node(run_dir, state, node="plan", note="Planning completed.")

    _advance_node(run_dir, state, node="prepare", agent="executor", note="Preparing official run state and adapter contracts.")
    _append_decision(run_dir, "executor", "prepare_complete", {"repo_root": str(repo_root)})
    _finish_node(run_dir, state, node="prepare", note="Preparation completed.")

    if dry_run:
        state["status"] = "planned"
        state["controller_status"] = "prepared"
        state["result_classification"] = "dry_run"
        state["next_action"] = "execute_manifest_without_dry_run"
        _append_decision(run_dir, "codex", "dry_run_complete", {"next_action": state["next_action"]})
        _write_json(run_dir / "analysis.json", _dry_run_analysis(state, manifest))
        _persist_state(run_dir, state)
        _append_index(run_dir.parent, _index_record(state))
        _refresh_dispatch_views(run_dir, state, manifest, registry)
        return run_dir

    execution_error: Exception | None = None
    try:
        state["status"] = "running"
        state["controller_status"] = "executing"
        _persist_state(run_dir, state)
        _append_event(run_dir, {"event": "run_started"})

        _advance_node(run_dir, state, node="execute", agent="executor", note="Starting compatibility stage execution.")
        for stage in state["stages"]:
            if stage["status"] == "skipped":
                continue
            _ensure_dependencies_satisfied(stage, state)
            _run_stage(stage, manifest, repo_root, run_dir, state)
            _persist_state(run_dir, state)
        _finish_node(run_dir, state, node="execute", note="All selected stages completed.")
    except Exception as exc:  # noqa: BLE001
        execution_error = exc
        state["status"] = "failed"
        state["failure_reason"] = str(exc)
        state["controller_status"] = "failed"
        _fail_node(run_dir, state, node="execute", note=str(exc))
        _append_event(run_dir, {"event": "run_failed", "error": str(exc)})
        _append_decision(run_dir, "executor", "execution_failed", {"error": str(exc)})
        _persist_state(run_dir, state)

    _advance_node(run_dir, state, node="monitor", agent="monitor", note="Summarizing compatibility execution state.")
    _append_decision(run_dir, "monitor", "stage_summary", {"stages": _stage_status_map(state)})
    _finish_node(run_dir, state, node="monitor", note="Monitoring summary recorded.")

    _advance_node(run_dir, state, node="analyze", agent="analyst", note="Classifying result and comparing against memory.")
    analysis = _build_analysis(state, manifest, knowledge_context, run_dir, execution_error)
    _write_json(run_dir / "analysis.json", analysis)
    state["result_classification"] = analysis["result_classification"]
    state["baseline_comparison"] = analysis["baseline_comparison"]
    state["next_action"] = analysis["next_action"]
    state["controller_status"] = "analyzed"
    _append_decision(run_dir, "analyst", "analysis_complete", analysis)
    _finish_node(run_dir, state, node="analyze", note=analysis["summary"])

    diagnosis = _default_placeholder("diagnosis")
    if _should_diagnose(manifest, analysis):
        _advance_node(run_dir, state, node="diagnose", agent="analyst", note="Generating diagnosis output.")
        diagnosis = _build_diagnosis(state, manifest, run_dir)
        _write_json(run_dir / "diagnosis.json", diagnosis)
        _append_decision(run_dir, "analyst", "diagnosis_complete", diagnosis)
        _finish_node(run_dir, state, node="diagnose", note=diagnosis["summary"])
        if state["status"] == "failed":
            state["status"] = "diagnosed"
            state["controller_status"] = "diagnosed"
    else:
        _skip_node(run_dir, state, node="diagnose", note="Diagnosis policy skipped.")
        _write_json(run_dir / "diagnosis.json", diagnosis)

    patch_request = _default_placeholder("patch_request")
    if _should_patch(manifest, analysis):
        _advance_node(run_dir, state, node="patch", agent="maintainer", note="Generating patch intent.")
        patch_request = _build_patch_request(state, manifest, diagnosis)
        _write_json(run_dir / "patch_request.json", patch_request)
        _append_decision(run_dir, "maintainer", "patch_request_generated", patch_request)
        _finish_node(run_dir, state, node="patch", note=patch_request["summary"])
    else:
        _skip_node(run_dir, state, node="patch", note="Patch policy skipped.")
        _write_json(run_dir / "patch_request.json", patch_request)

    next_manifest = _default_placeholder("next_manifest")
    if _should_generate_next(manifest):
        _advance_node(run_dir, state, node="propose_next", agent="planner", note="Generating follow-up manifest proposal.")
        next_manifest = _build_next_manifest(state, manifest, analysis, diagnosis)
        _write_json(run_dir / "next_manifest.json", next_manifest)
        _append_decision(run_dir, "planner", "next_manifest_generated", {"experiment_id": next_manifest.get("experiment_id")})
        _finish_node(run_dir, state, node="propose_next", note="Follow-up manifest proposal created.")
    else:
        _skip_node(run_dir, state, node="propose_next", note="Next-step policy skipped.")
        _write_json(run_dir / "next_manifest.json", next_manifest)

    if _auto_run_enabled(state, manifest):
        _advance_node(run_dir, state, node="queue_next", agent="codex", note="Queueing follow-up manifest proposal.")
        state["spawned_runs"].append({
            "proposed_experiment_id": next_manifest.get("experiment_id"),
            "status": "proposed",
            "source_run_id": state["run_id"],
        })
        state["next_action"] = "followup_manifest_queued"
        _append_decision(run_dir, "codex", "followup_queued", {"experiment_id": next_manifest.get("experiment_id")})
        _append_action(run_dir, "codex", "auto_queue_followup", {"experiment_id": next_manifest.get("experiment_id")})
        _finish_node(run_dir, state, node="queue_next", note="Follow-up proposal queued in controller state.")
    else:
        _skip_node(run_dir, state, node="queue_next", note="Auto-queue disabled by policy.")

    if state["status"] not in {"failed", "diagnosed"}:
        state["status"] = "completed"
        state["controller_status"] = "completed"
        state["finished_at"] = utc_now()

    _persist_state(run_dir, state)
    update_knowledge(repo_root, state, analysis, manifest)
    state["knowledge_updated"] = True
    _append_index(run_dir.parent, _index_record(state))
    _refresh_dispatch_views(run_dir, state, manifest, registry)

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
    _append_decision(run_dir, "codex", "resume_requested", {"selected_stages": selected, "dry_run": dry_run})
    if dry_run:
        return run_dir

    state["status"] = "running"
    state["controller_status"] = "resumed"
    _persist_state(run_dir, state)

    execution_error: Exception | None = None
    try:
        _advance_node(run_dir, state, node="execute", agent="executor", note="Resuming pending stages.")
        for stage in state["stages"]:
            if stage["name"] not in selected:
                continue
            _ensure_dependencies_satisfied(stage, state)
            _run_stage(stage, manifest, repo_root, run_dir, state)
            _persist_state(run_dir, state)
        _finish_node(run_dir, state, node="execute", note="Resumed stages completed.")
    except Exception as exc:  # noqa: BLE001
        execution_error = exc
        state["status"] = "failed"
        state["failure_reason"] = str(exc)
        _fail_node(run_dir, state, node="execute", note=str(exc))
        _persist_state(run_dir, state)

    _append_index(run_dir.parent, _index_record(state))
    if execution_error is not None:
        raise RuntimeErrorWithContext(str(execution_error))
    return run_dir


def summarize_run(run_dir_arg: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir_arg).resolve()
    state = _read_json(run_dir / "state.json")
    analysis = _read_optional_json(run_dir / "analysis.json")
    diagnosis = _read_optional_json(run_dir / "diagnosis.json")
    return {
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "status": state["status"],
        "controller_status": state.get("controller_status"),
        "dispatch_mode": state.get("dispatch_mode"),
        "queue_owner": state.get("queue_owner"),
        "active_agent": state.get("active_agent"),
        "result_classification": state.get("result_classification"),
        "next_action": state.get("next_action"),
        "failure_reason": state.get("failure_reason"),
        "stages": _stage_status_map(state),
        "dispatch": {
            "active_packet_ids": state.get("active_packet_ids", []),
            "awaiting_results_from": state.get("awaiting_results_from", []),
            "accepted_results": state.get("accepted_results", []),
            "rejected_results": state.get("rejected_results", []),
            "reported_results": state.get("reported_results", []),
        },
        "analysis": analysis,
        "diagnosis": diagnosis,
    }


def _create_run_context(
    *,
    manifest: Dict[str, Any],
    manifest_path: Path,
    repo_root: Path,
    registry: Dict[str, Any],
    knowledge_context: Dict[str, Any],
    run_id: str | None,
    from_stage: str | None,
    through_stage: str | None,
    dry_run: bool,
    dispatch_mode: str,
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
        manifest,
        manifest_path,
        repo_root,
        run_dir,
        resolved_run_id,
        selected,
        registry,
        knowledge_context,
        dry_run=dry_run,
        dispatch_mode=dispatch_mode,
    )
    return run_dir, state


def _load_run_context(run_dir_arg: str | Path) -> tuple[Path, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    run_dir = Path(run_dir_arg).resolve()
    state = _read_json(run_dir / "state.json")
    manifest = _read_json(run_dir / "manifest.snapshot.json")
    repo_root = Path(state["workspace_root"]).resolve()
    registry = load_agent_registry(repo_root)
    knowledge_context = load_knowledge_context(repo_root, manifest["family"])
    return run_dir, state, manifest, registry, knowledge_context


def _initial_state(
    manifest: Dict[str, Any],
    manifest_path: Path,
    repo_root: Path,
    run_dir: Path,
    run_id: str,
    selected: List[str],
    registry: Dict[str, Any],
    knowledge_context: Dict[str, Any],
    *,
    dry_run: bool,
    dispatch_mode: str,
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

    agents = bootstrap_agent_state(registry)
    state = {
        "run_id": run_id,
        "experiment_id": manifest["experiment_id"],
        "manifest_path": str(manifest_path),
        "workspace_root": str(repo_root),
        "run_dir": str(run_dir),
        "authority_source": manifest.get("autonomy", {}).get("authority_source", registry.get("authority_source", "AGENTS.md")),
        "autonomy_mode": manifest.get("autonomy", {}).get("mode", knowledge_context["default_policies"].get("autonomy_mode", "closed_loop")),
        "status": "planned",
        "controller_status": "planning",
        "dry_run": dry_run,
        "dispatch_mode": dispatch_mode,
        "queue_owner": registry["top_level_controller"],
        "active_agent": registry["top_level_controller"],
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "failure_reason": None,
        "decision_log_ref": str((run_dir / "decision_log.jsonl").resolve()),
        "agent_handoffs_ref": str((run_dir / "agent_handoffs.jsonl").resolve()),
        "analysis_ref": str((run_dir / "analysis.json").resolve()),
        "diagnosis_ref": str((run_dir / "diagnosis.json").resolve()),
        "patch_request_ref": str((run_dir / "patch_request.json").resolve()),
        "next_manifest_ref": str((run_dir / "next_manifest.json").resolve()),
        "dispatch_plan_ref": str((run_dir / "dispatch_plan.json").resolve()),
        "controller_actions_ref": str((run_dir / "controller_actions.jsonl").resolve()),
        "agent_packets_dir": str((run_dir / "agent_packets").resolve()),
        "agent_results_dir": str((run_dir / "agent_results").resolve()),
        "hypothesis": manifest["hypothesis"],
        "baseline_refs": manifest.get("baseline_refs", []),
        "baseline_comparison": _baseline_stub(knowledge_context),
        "result_classification": "planned",
        "next_action": None,
        "supersedes": manifest.get("supersedes"),
        "spawned_runs": [],
        "awaiting_results_from": [],
        "active_packet_ids": [],
        "reported_results": [],
        "accepted_results": [],
        "rejected_results": [],
        "superseded_results": [],
        "knowledge_updated": False,
        "auto_patch_enabled": _auto_patch_enabled(manifest, knowledge_context),
        "auto_run_enabled": _auto_run_enabled_from_policy(manifest, knowledge_context),
        "agents": agents,
        "controller_graph": build_controller_graph(manifest),
        "stages": stages,
    }
    enrich_controller_graph(state, registry)
    set_active_agent(state, registry["top_level_controller"])
    return state


def _initialize_run_artifacts(run_dir: Path, manifest: Dict[str, Any], registry: Dict[str, Any], state: Dict[str, Any]) -> None:
    _write_json(run_dir / "manifest.snapshot.json", manifest)
    _write_json(run_dir / "analysis.json", _default_placeholder("analysis"))
    _write_json(run_dir / "diagnosis.json", _default_placeholder("diagnosis"))
    _write_json(run_dir / "patch_request.json", _default_placeholder("patch_request"))
    _write_json(run_dir / "next_manifest.json", _default_placeholder("next_manifest"))
    _write_json(run_dir / "dispatch_plan.json", build_dispatch_plan(state, manifest, registry))
    (run_dir / "decision_log.jsonl").write_text("", encoding="utf-8")
    (run_dir / "agent_handoffs.jsonl").write_text("", encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "controller_actions.jsonl").write_text("", encoding="utf-8")
    (run_dir / "agent_packets").mkdir(parents=True, exist_ok=True)
    (run_dir / "agent_results").mkdir(parents=True, exist_ok=True)


def _refresh_dispatch_views(run_dir: Path, state: Dict[str, Any], manifest: Dict[str, Any], registry: Dict[str, Any]) -> None:
    _write_json(run_dir / "dispatch_plan.json", build_dispatch_plan(state, manifest, registry))


def _mark_node_ready(state: Dict[str, Any], node_name: str, *, reason: str) -> None:
    node = graph_node_by_name(state, node_name)
    if node.get("dispatch_status") in {"accepted", "delegated", "reported", "superseded"}:
        return
    node["dispatch_status"] = "ready"
    if node["status"] == "planned":
        node["notes"].append(reason)


def _apply_node_result(
    run_dir: Path,
    state: Dict[str, Any],
    manifest: Dict[str, Any],
    knowledge_context: Dict[str, Any],
    node: Dict[str, Any],
    result: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    agent = node["agent"]
    state["agents"][agent]["last_output"] = result.get("summary")
    update_graph_node(
        state,
        node["node"],
        status=_node_status_from_result(node["node"], result, payload),
        timestamp=utc_now(),
        note=result.get("summary") or f"Accepted packet result for {node['node']}.",
    )
    set_active_agent(state, state["queue_owner"], node=node["node"])

    if node["node"] == "plan":
        state["controller_status"] = "planned"
        _mark_node_ready(state, "prepare", reason="Planner result accepted.")
    elif node["node"] == "prepare":
        state["controller_status"] = "prepared"
        _mark_node_ready(state, "execute", reason="Preparation accepted; ready for execution packet.")
    elif node["node"] == "execute":
        stage_statuses = payload.get("stage_statuses", {})
        if stage_statuses:
            for stage in state["stages"]:
                if stage["name"] in stage_statuses:
                    stage["status"] = stage_statuses[stage["name"]]
        if result.get("status", "").lower() in {"failed", "error"}:
            state["status"] = "failed"
            state["failure_reason"] = result.get("summary") or payload.get("failure_reason") or "Execution packet reported failure."
        _mark_node_ready(state, "monitor", reason="Execution result accepted.")
    elif node["node"] == "monitor":
        _mark_node_ready(state, "analyze", reason="Monitor report accepted.")
    elif node["node"] == "analyze":
        analysis = payload.get("analysis") or payload or _build_analysis(state, manifest, knowledge_context, run_dir, None)
        if analysis.get("status") != "generated":
            analysis["status"] = "generated"
            analysis["generated_at"] = utc_now()
            analysis.setdefault("run_id", state["run_id"])
            analysis.setdefault("experiment_id", state["experiment_id"])
        _write_json(run_dir / "analysis.json", analysis)
        state["result_classification"] = analysis.get("result_classification", state.get("result_classification"))
        state["baseline_comparison"] = analysis.get("baseline_comparison", state.get("baseline_comparison"))
        state["next_action"] = analysis.get("next_action")
        state["controller_status"] = "analyzed"
        if state["result_classification"] in {"failed", "needs_iteration"}:
            if manifest.get("diagnosis_policy", {}).get("mode", "none") != "none":
                _mark_node_ready(state, "diagnose", reason="Analysis requires diagnosis.")
            elif _should_generate_next(manifest) or state.get("auto_run_enabled"):
                _mark_node_ready(state, "propose_next", reason="Diagnosis skipped; ready to propose follow-up.")
        elif _should_generate_next(manifest) or state.get("auto_run_enabled"):
            _mark_node_ready(state, "propose_next", reason="Analysis accepted; ready to propose follow-up.")
    elif node["node"] == "diagnose":
        diagnosis = payload.get("diagnosis") or payload or _build_diagnosis(state, manifest, run_dir)
        if diagnosis.get("status") != "generated":
            diagnosis["status"] = "generated"
            diagnosis["generated_at"] = utc_now()
            diagnosis.setdefault("run_id", state["run_id"])
        _write_json(run_dir / "diagnosis.json", diagnosis)
        state["controller_status"] = "diagnosed"
        if manifest.get("patch_policy", {}).get("mode", "none") != "none":
            _mark_node_ready(state, "patch", reason="Diagnosis accepted; ready for patch packet.")
        elif _should_generate_next(manifest) or state.get("auto_run_enabled"):
            _mark_node_ready(state, "propose_next", reason="Patch skipped; ready to propose follow-up.")
    elif node["node"] == "patch":
        patch_request = payload.get("patch_request") or payload or _build_patch_request(state, manifest, _read_json(run_dir / "diagnosis.json"))
        if patch_request.get("status") not in {"requested", "generated", "applied"}:
            patch_request["status"] = "requested"
            patch_request["generated_at"] = utc_now()
            patch_request.setdefault("run_id", state["run_id"])
        if state.get("auto_patch_enabled"):
            patch_request["status"] = "applied"
            _append_action(run_dir, state["queue_owner"], "auto_patch_apply", {"node": node["node"], "packet_id": node["packet_id"]})
        _write_json(run_dir / "patch_request.json", patch_request)
        if _should_generate_next(manifest) or state.get("auto_run_enabled"):
            _mark_node_ready(state, "propose_next", reason="Patch accepted; ready to propose follow-up.")
    elif node["node"] == "propose_next":
        next_manifest = payload.get("next_manifest") or payload or _build_next_manifest(state, manifest, _read_json(run_dir / "analysis.json"), _read_json(run_dir / "diagnosis.json"))
        _write_json(run_dir / "next_manifest.json", next_manifest)
        if state.get("auto_run_enabled") or manifest.get("next_step_policy", {}).get("auto_queue", False):
            _mark_node_ready(state, "queue_next", reason="Follow-up manifest accepted; ready to queue next run.")
    elif node["node"] == "queue_next":
        queued = payload.get("queued_followup") or {
            "proposed_experiment_id": (_read_json(run_dir / "next_manifest.json")).get("experiment_id"),
            "status": "queued",
            "source_run_id": state["run_id"],
        }
        state["spawned_runs"].append(queued)
        state["next_action"] = "followup_manifest_queued"
        _append_action(run_dir, state["queue_owner"], "auto_queue_followup", queued)


def _node_status_from_result(node_name: str, result: Dict[str, Any], payload: Dict[str, Any]) -> str:
    explicit = payload.get("node_status")
    if explicit in {"completed", "failed", "skipped", "running", "planned"}:
        return explicit
    status = str(result.get("status", "")).lower()
    if status in {"failed", "error"} and node_name == "execute":
        return "failed"
    return "completed"


def _node_dispatch_status(state: Dict[str, Any], node_name: str) -> str | None:
    try:
        return graph_node_by_name(state, node_name).get("dispatch_status")
    except KeyError:
        return None


def _auto_patch_enabled(manifest: Dict[str, Any], knowledge_context: Dict[str, Any]) -> bool:
    patch_policy = manifest.get("patch_policy", {})
    if "auto_apply" in patch_policy:
        return bool(patch_policy.get("auto_apply"))
    return bool(knowledge_context["default_policies"].get("auto_patch_apply", True))


def _auto_run_enabled_from_policy(manifest: Dict[str, Any], knowledge_context: Dict[str, Any]) -> bool:
    next_step_policy = manifest.get("next_step_policy", {})
    if "auto_queue" in next_step_policy:
        return bool(next_step_policy.get("auto_queue"))
    return bool(knowledge_context["default_policies"].get("auto_run_dispatch", True))


def _auto_run_enabled(state: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
    if state.get("dispatch_mode") == "native_multi_agent":
        return state.get("auto_run_enabled", False)
    return bool(state.get("auto_run_enabled", False))


def _advance_node(run_dir: Path, state: Dict[str, Any], *, node: str, agent: str, note: str) -> None:
    previous = state.get("active_agent")
    timestamp = utc_now()
    update_graph_node(state, node, status="running", timestamp=timestamp, note=note)
    set_active_agent(state, agent, node=node)
    state["controller_status"] = node
    state["updated_at"] = timestamp
    _append_handoff(run_dir, previous, agent, node=node, reason=note)
    _persist_state(run_dir, state)


def _finish_node(run_dir: Path, state: Dict[str, Any], *, node: str, note: str) -> None:
    update_graph_node(state, node, status="completed", timestamp=utc_now(), note=note)
    state["updated_at"] = utc_now()
    _persist_state(run_dir, state)


def _fail_node(run_dir: Path, state: Dict[str, Any], *, node: str, note: str) -> None:
    update_graph_node(state, node, status="failed", timestamp=utc_now(), note=note)
    _persist_state(run_dir, state)


def _skip_node(run_dir: Path, state: Dict[str, Any], *, node: str, note: str) -> None:
    update_graph_node(state, node, status="skipped", timestamp=utc_now(), note=note)
    _persist_state(run_dir, state)


def _should_diagnose(manifest: Dict[str, Any], analysis: Dict[str, Any]) -> bool:
    mode = manifest.get("diagnosis_policy", {}).get("mode", "none")
    if mode == "none":
        return False
    return analysis["result_classification"] in {"failed", "needs_iteration"}


def _should_patch(manifest: Dict[str, Any], analysis: Dict[str, Any]) -> bool:
    mode = manifest.get("patch_policy", {}).get("mode", "none")
    if mode == "none":
        return False
    return analysis["result_classification"] in {"failed", "needs_iteration"}


def _should_generate_next(manifest: Dict[str, Any]) -> bool:
    return manifest.get("next_step_policy", {}).get("auto_generate_manifest", False)


def _dry_run_analysis(state: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "generated",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "result_classification": "dry_run",
        "summary": "Dry-run completed. Manifest and controller state are ready for execution.",
        "baseline_comparison": state["baseline_comparison"],
        "unmet_reasons": [],
        "suggested_changes": [],
        "next_action": "execute_manifest_without_dry_run",
    }


def _build_analysis(state: Dict[str, Any], manifest: Dict[str, Any], knowledge_context: Dict[str, Any], run_dir: Path, execution_error: Exception | None) -> Dict[str, Any]:
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

    if execution_error is not None:
        classification = "failed"
    elif unmet_reasons:
        classification = "needs_iteration"
    else:
        classification = acceptance.get("result_classification_on_pass") or manifest.get("promotion_policy", {}).get("default_classification_on_pass") or knowledge_context["default_policies"].get("default_result_classification_on_pass", "candidate")

    baseline = _baseline_stub(knowledge_context)
    if execution_error is not None:
        summary = f"Run failed during execution: {execution_error}"
        next_action = manifest.get("next_step_policy", {}).get("on_failure") or knowledge_context["default_policies"].get("default_failure_action")
    elif unmet_reasons:
        summary = "Run completed execution but did not satisfy acceptance criteria."
        next_action = manifest.get("next_step_policy", {}).get("on_unmet_criteria") or knowledge_context["default_policies"].get("default_unmet_action")
    else:
        summary = "Run satisfied controller acceptance criteria."
        next_action = manifest.get("next_step_policy", {}).get("on_success") or knowledge_context["default_policies"].get("default_success_action")

    return {
        "status": "generated",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "result_classification": classification,
        "summary": summary,
        "baseline_comparison": baseline,
        "unmet_reasons": unmet_reasons,
        "suggested_changes": _suggested_changes_from_failures(state),
        "next_action": next_action,
    }


def _build_diagnosis(state: Dict[str, Any], manifest: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    suggestions = _suggested_changes_from_failures(state)
    summary = state.get("failure_reason") or "Acceptance criteria were unmet without a hard execution failure."
    return {
        "status": "generated",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "summary": summary,
        "suspected_causes": suggestions or ["Inspect stage logs and compare against known failure signatures."],
        "policy_mode": manifest.get("diagnosis_policy", {}).get("mode", "unspecified"),
    }


def _build_patch_request(state: Dict[str, Any], manifest: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    allowed_paths = manifest.get("patch_policy", {}).get("allowed_paths", [])
    return {
        "status": "requested",
        "generated_at": utc_now(),
        "run_id": state["run_id"],
        "summary": "Generate a targeted patch proposal from the diagnosis output.",
        "allowed_paths": allowed_paths,
        "reason": diagnosis.get("summary"),
        "change_intent": diagnosis.get("suspected_causes", []),
    }


def _build_next_manifest(state: Dict[str, Any], manifest: Dict[str, Any], analysis: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    followup = json.loads(json.dumps(manifest))
    followup["experiment_id"] = f"{manifest['experiment_id']}_followup"
    followup["baseline_refs"] = list(dict.fromkeys([state["run_id"], *manifest.get("baseline_refs", [])]))
    followup["hypothesis"] = f"Follow-up to {state['run_id']}: {analysis['next_action']}"
    followup["lineage"] = {
        "parent_run_id": state["run_id"],
        "source_experiment_id": state["experiment_id"],
        "previous_result_classification": analysis["result_classification"],
    }
    followup["controller_note"] = {
        "generated_from": state["run_id"],
        "next_action": analysis["next_action"],
        "diagnosis_summary": diagnosis.get("summary"),
    }
    return followup


def _baseline_stub(knowledge_context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family": knowledge_context["family"],
        "best_run": knowledge_context.get("best_run"),
        "family_summary": knowledge_context.get("family_experiments", {}).get("summary"),
    }


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
    set_active_agent(state, "executor", node="execute")
    _append_event(run_dir, {"event": "stage_started", "stage": stage_state["name"], "command": command, "cwd": str(cwd)})
    _append_decision(run_dir, "executor", "stage_started", {"stage": stage_state["name"], "command": command})

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
        _append_event(run_dir, {"event": "stage_failed", "stage": stage_state["name"], "returncode": completed.returncode, "log_path": str(log_path)})
        raise RuntimeErrorWithContext(f"Stage {stage_state['name']} failed with return code {completed.returncode}. See {log_path}")

    _check_completion(stage_def, repo_root, run_dir, log_path, stage_state["name"])
    stage_state["status"] = "completed"
    _append_event(run_dir, {"event": "stage_completed", "stage": stage_state["name"], "log_path": str(log_path)})
    _append_decision(run_dir, "monitor", "stage_completed", {"stage": stage_state["name"], "log_path": str(log_path)})


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
        "controller_status": state.get("controller_status"),
        "dispatch_mode": state.get("dispatch_mode"),
        "result_classification": state.get("result_classification"),
        "manifest_path": state["manifest_path"],
    }


def _append_event(run_dir: Path, event: Dict[str, Any]) -> None:
    payload = {"ts": utc_now(), **event}
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_decision(run_dir: Path, agent: str, decision: str, payload: Dict[str, Any]) -> None:
    record = {"ts": utc_now(), "agent": agent, "decision": decision, "payload": payload}
    with (run_dir / "decision_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_handoff(run_dir: Path, from_agent: str | None, to_agent: str, *, node: str, reason: str) -> None:
    record = {"ts": utc_now(), "from_agent": from_agent, "to_agent": to_agent, "node": node, "reason": reason}
    with (run_dir / "agent_handoffs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_action(run_dir: Path, actor: str, action: str, payload: Dict[str, Any]) -> None:
    record = {"ts": utc_now(), "actor": actor, "action": action, "payload": payload}
    with (run_dir / "controller_actions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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

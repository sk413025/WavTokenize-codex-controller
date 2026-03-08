from __future__ import annotations

from pathlib import Path

from codex_controller.runtime import run_manifest, summarize_run


def test_dry_run_creates_minimal_runtime_contract(patch_runtime_manifest, read_json) -> None:
    manifest_path, _ = patch_runtime_manifest("experiments/manifests/exp0304_material_generalization_preflight.json")
    run_dir = run_manifest(manifest_path, run_id="minimal_contract", dry_run=True)

    state = read_json(run_dir / "state.json")
    assert state["run_status_detail"] == "prepared"
    assert state["result_classification"] == "dry_run"

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "analysis.json").exists()
    assert (run_dir / "diagnosis.json").exists()
    assert (run_dir / "monitor_report.json").exists()
    assert (run_dir / "events.jsonl").exists()

    assert not (run_dir / "agent_packets").exists()
    assert not (run_dir / "agent_results").exists()
    assert not (run_dir / "dispatch_plan.json").exists()
    assert not (run_dir / "decision_log.jsonl").exists()
    assert not (run_dir / "agent_handoffs.jsonl").exists()
    assert not (run_dir / "controller_actions.jsonl").exists()


def test_status_summary_omits_dispatch_fields(patch_runtime_manifest) -> None:
    manifest_path, _ = patch_runtime_manifest("experiments/manifests/exp0304_material_generalization_preflight.json")
    run_dir = run_manifest(manifest_path, run_id="minimal_status", dry_run=True)

    summary = summarize_run(run_dir)
    assert "dispatch_mode" not in summary
    assert "queue_owner" not in summary
    assert "dispatch" not in summary



def test_status_summary_reads_legacy_controller_status(patch_runtime_manifest, read_json) -> None:
    manifest_path, _ = patch_runtime_manifest("experiments/manifests/exp0304_material_generalization_preflight.json")
    run_dir = run_manifest(manifest_path, run_id="legacy_status", dry_run=True)

    state = read_json(run_dir / "state.json")
    state.pop("run_status_detail", None)
    state["controller_status"] = "prepared"
    (run_dir / "state.json").write_text(__import__("json").dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = summarize_run(run_dir)
    assert summary["run_status_detail"] == "prepared"

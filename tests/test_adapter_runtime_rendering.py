from __future__ import annotations

from pathlib import Path

from codex_controller.adapters import resolve_stage
from codex_controller.runtime import _build_command, _check_completion


def test_resolve_stage_and_render_run_dir(
    official_preflight_manifest: tuple[Path, dict[str, object]],
    repo_root: Path,
    tmp_path: Path,
) -> None:
    _, manifest = official_preflight_manifest
    stage = manifest["stages"][0]
    resolved = resolve_stage(stage, repo_root)
    run_dir = tmp_path / "controller_runs" / "render_case"
    run_dir.mkdir(parents=True)

    command = _build_command(resolved, repo_root, run_dir)

    assert command[0].endswith("python") or command[0].endswith("python3")
    assert command[1].endswith("exp_0304/preflight_material_gen.py")
    assert "--report_path" in command
    report_arg = command[command.index("--report_path") + 1]
    assert report_arg == str(run_dir / "preflight_report.json")
    assert resolved["completion"]["file_exists"] == ["{run_dir}/preflight_report.json"]


def test_completion_check_renders_run_dir_placeholder(
    official_preflight_manifest: tuple[Path, dict[str, object]],
    repo_root: Path,
    tmp_path: Path,
) -> None:
    _, manifest = official_preflight_manifest
    stage = resolve_stage(manifest["stages"][0], repo_root)
    run_dir = tmp_path / "controller_runs" / "completion_case"
    run_dir.mkdir(parents=True)
    (run_dir / "preflight_report.json").write_text("{}\n", encoding="utf-8")
    log_path = run_dir / "preflight_material_gen.log"
    log_path.write_text("PREFLIGHT STATUS: ready_for_smoke\n", encoding="utf-8")

    _check_completion(stage, repo_root, run_dir, log_path, "preflight_material_gen")

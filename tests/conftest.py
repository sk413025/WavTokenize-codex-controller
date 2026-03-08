from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from codex_controller.manifest import load_manifest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def official_preflight_manifest(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path, manifest, _ = load_manifest(
        repo_root / "experiments/manifests/exp0304_material_generalization_preflight.json"
    )
    return manifest_path, manifest


@pytest.fixture
def patch_runtime_manifest(monkeypatch: pytest.MonkeyPatch, repo_root: Path, tmp_path: Path):
    def _patch(manifest_relpath: str) -> tuple[Path, dict[str, Any]]:
        manifest_path, manifest, _ = load_manifest(repo_root / manifest_relpath)
        manifest = copy.deepcopy(manifest)
        manifest["run_root"] = str(tmp_path / "controller_runs")

        def fake_load_manifest(_: str | Path):
            return manifest_path, manifest, repo_root

        import codex_controller.runtime as runtime

        monkeypatch.setattr(runtime, "load_manifest", fake_load_manifest)
        return manifest_path, manifest

    return _patch


@pytest.fixture
def read_json():
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return _read_json

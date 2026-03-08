from __future__ import annotations

import copy
from pathlib import Path

import pytest

from codex_controller.manifest import ManifestError, load_manifest, validate_manifest


def test_official_manifest_loads_and_validates(repo_root: Path) -> None:
    manifest_path, manifest, resolved_repo_root = load_manifest(
        repo_root / "experiments/manifests/exp0304_material_generalization_preflight.json"
    )

    assert manifest_path.name == "exp0304_material_generalization_preflight.json"
    assert manifest["experiment_id"] == "exp0304_material_generalization_preflight"
    assert resolved_repo_root == repo_root
    assert manifest["stages"][0]["adapter_id"] == "exp0304_preflight_material_gen"


def test_validate_manifest_rejects_unknown_stage_dependency(
    official_preflight_manifest: tuple[Path, dict[str, object]],
    repo_root: Path,
) -> None:
    _, manifest = official_preflight_manifest
    broken = copy.deepcopy(manifest)
    broken["stages"][0]["after"] = ["missing_stage"]

    with pytest.raises(ManifestError, match="depends on unknown stage"):
        validate_manifest(broken, repo_root=repo_root)

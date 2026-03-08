from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .adapters import adapter_exists

SUPPORTED_ADAPTERS = {"python_script", "shell"}
REQUIRED_TOP_LEVEL = [
    "schema_version",
    "experiment_id",
    "family",
    "objective",
    "hypothesis",
    "acceptance_criteria",
    "stages",
]


class ManifestError(ValueError):
    """Raised when a manifest is malformed."""


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise ManifestError(f"Could not locate repo root from {start}")


def load_manifest(path: str | Path) -> tuple[Path, Dict[str, Any], Path]:
    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise ManifestError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    repo_root = find_repo_root(manifest_path.parent)
    validate_manifest(manifest, repo_root=repo_root, manifest_path=manifest_path)
    return manifest_path, manifest, repo_root


def validate_manifest(manifest: Dict[str, Any], *, repo_root: Path, manifest_path: Path | None = None) -> None:
    for key in REQUIRED_TOP_LEVEL:
        if key not in manifest:
            raise ManifestError(_prefix(manifest_path) + f"Missing required key: {key}")

    if manifest["schema_version"] != 1:
        raise ManifestError(_prefix(manifest_path) + "Only schema_version=1 is supported")

    if not isinstance(manifest.get("hypothesis"), str) or not manifest["hypothesis"].strip():
        raise ManifestError(_prefix(manifest_path) + "hypothesis must be a non-empty string")

    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ManifestError(_prefix(manifest_path) + "stages must be a non-empty list")

    names: List[str] = []
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ManifestError(_prefix(manifest_path) + f"Stage #{idx} must be an object")
        for key in ("name", "adapter"):
            if key not in stage:
                raise ManifestError(_prefix(manifest_path) + f"Stage #{idx} missing required key: {key}")
        stage_name = stage["name"]
        if not isinstance(stage_name, str) or not stage_name:
            raise ManifestError(_prefix(manifest_path) + f"Stage #{idx} name must be a non-empty string")
        if stage_name in names:
            raise ManifestError(_prefix(manifest_path) + f"Duplicate stage name: {stage_name}")
        names.append(stage_name)

        adapter = stage["adapter"]
        if adapter not in SUPPORTED_ADAPTERS:
            raise ManifestError(_prefix(manifest_path) + f"Unsupported adapter '{adapter}' in stage {stage_name}")

        adapter_id = stage.get("adapter_id")
        if adapter_id and not adapter_exists(repo_root, adapter_id):
            raise ManifestError(_prefix(manifest_path) + f"Unknown adapter_id '{adapter_id}' in stage {stage_name}")

        if adapter == "python_script" and not (stage.get("entrypoint") or adapter_id):
            raise ManifestError(_prefix(manifest_path) + f"Stage {stage_name} requires 'entrypoint' or 'adapter_id'")
        if adapter == "shell" and not (stage.get("command") or adapter_id):
            raise ManifestError(_prefix(manifest_path) + f"Stage {stage_name} requires 'command' or 'adapter_id'")

        after = stage.get("after", [])
        if after is None:
            after = []
        if not isinstance(after, list):
            raise ManifestError(_prefix(manifest_path) + f"Stage {stage_name} field 'after' must be a list")

    known = set(names)
    for stage in stages:
        for dependency in stage.get("after", []) or []:
            if dependency not in known:
                raise ManifestError(
                    _prefix(manifest_path)
                    + f"Stage {stage['name']} depends on unknown stage '{dependency}'"
                )


def describe_manifest(manifest: Dict[str, Any]) -> str:
    lines = [
        f"experiment_id: {manifest['experiment_id']}",
        f"family: {manifest['family']}",
        f"objective: {manifest['objective']}",
        f"hypothesis: {manifest['hypothesis']}",
        f"run_root: {manifest.get('run_root', 'controller_runs')}",
        f"autonomy: {manifest.get('autonomy', {}).get('mode', 'unspecified')}",
        "stages:",
    ]
    for stage in manifest["stages"]:
        after = ", ".join(stage.get("after", [])) or "-"
        adapter_id = stage.get("adapter_id", "-")
        lines.append(f"  - {stage['name']} [{stage['adapter']}] adapter_id={adapter_id} after={after}")
    return "\n".join(lines)


def _prefix(path: Path | None) -> str:
    return f"{path}: " if path else ""

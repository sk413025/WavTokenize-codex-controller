from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


class AdapterError(ValueError):
    """Raised when an adapter contract is missing or malformed."""


ADAPTER_INDEX_PATH = Path("experiments/adapters/index.json")


def load_adapter_index(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / ADAPTER_INDEX_PATH
    if not path.exists():
        raise AdapterError(f"Adapter index does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def adapter_exists(repo_root: Path, adapter_id: str) -> bool:
    try:
        load_adapter(repo_root, adapter_id)
    except AdapterError:
        return False
    return True


def load_adapter(repo_root: Path, adapter_id: str) -> Dict[str, Any]:
    index = load_adapter_index(repo_root)
    for adapter in index.get("adapters", []):
        if adapter.get("adapter_id") == adapter_id:
            adapter_path = repo_root / adapter["path"]
            if not adapter_path.exists():
                raise AdapterError(f"Adapter file does not exist: {adapter_path}")
            with adapter_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("adapter_id") != adapter_id:
                raise AdapterError(f"Adapter id mismatch in {adapter_path}")
            return payload
    raise AdapterError(f"Unknown adapter id: {adapter_id}")


def resolve_stage(stage: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    resolved = deepcopy(stage)
    adapter_id = stage.get("adapter_id")
    if not adapter_id:
        return resolved

    adapter = load_adapter(repo_root, adapter_id)
    merged = deepcopy(adapter)
    merged.update(resolved)

    merged["adapter_id"] = adapter_id
    merged["known_failures"] = adapter.get("known_failures", [])
    completion = deepcopy(adapter.get("completion", {}))
    completion.update(resolved.get("completion", {}))
    merged["completion"] = completion

    artifacts = deepcopy(adapter.get("artifacts", {}))
    artifacts.update(resolved.get("artifacts", {}))
    merged["artifacts"] = artifacts

    if "adapter" not in merged and "mode" in adapter:
        merged["adapter"] = adapter["mode"]
    return merged

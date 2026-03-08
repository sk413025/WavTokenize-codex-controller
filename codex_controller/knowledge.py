from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

KNOWLEDGE_PATHS = {
    "experiments": Path("knowledge/experiments/index.json"),
    "failures": Path("knowledge/failures/index.json"),
    "policies": Path("knowledge/policies/controller_defaults.json"),
    "best_runs": Path("knowledge/best_runs.json"),
}


def load_knowledge_context(repo_root: Path, family: str) -> Dict[str, Any]:
    experiments = _read_json(repo_root / KNOWLEDGE_PATHS["experiments"])
    failures = _read_json(repo_root / KNOWLEDGE_PATHS["failures"])
    policies = _read_json(repo_root / KNOWLEDGE_PATHS["policies"])
    best_runs = _read_json(repo_root / KNOWLEDGE_PATHS["best_runs"])
    return {
        "family": family,
        "family_experiments": deepcopy(experiments.get("families", {}).get(family, {})),
        "family_failures": deepcopy(failures.get("families", {}).get(family, [])),
        "default_policies": deepcopy(policies),
        "best_run": deepcopy(best_runs.get("families", {}).get(family)),
    }


def update_knowledge(repo_root: Path, state: Dict[str, Any], analysis: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    family = manifest["family"]
    experiments_path = repo_root / KNOWLEDGE_PATHS["experiments"]
    failures_path = repo_root / KNOWLEDGE_PATHS["failures"]

    experiments = _read_json(experiments_path)
    failures = _read_json(failures_path)

    experiments.setdefault("families", {}).setdefault(
        family,
        {
            "summary": manifest["objective"],
            "latest_run_id": None,
            "latest_result_classification": None,
            "baseline_run_id": None,
            "notes": [],
            "paused": False,
            "last_next_action": None,
        },
    )
    family_entry = experiments["families"][family]
    family_entry["latest_run_id"] = state["run_id"]
    family_entry["latest_result_classification"] = analysis["result_classification"]
    family_entry["last_next_action"] = analysis.get("next_action")
    note = analysis.get("summary")
    if note and note not in family_entry["notes"]:
        family_entry["notes"].append(note)

    failures.setdefault("families", {}).setdefault(family, [])
    if analysis["result_classification"] == "failed":
        failures["families"][family].append(
            {
                "run_id": state["run_id"],
                "failure_reason": state.get("failure_reason"),
                "summary": analysis.get("summary"),
                "suggested_changes": analysis.get("suggested_changes", []),
            }
        )

    _write_json(experiments_path, experiments)
    _write_json(failures_path, failures)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

NODE_SKILLS = {
    "plan": "controller-decomposition",
    "prepare": "dispatch-handoff",
    "execute": "dispatch-handoff",
    "monitor": "stage-monitoring",
    "analyze": "run-diagnosis",
    "diagnose": "run-diagnosis",
    "patch": "dispatch-handoff",
    "propose_next": "followup-generation",
    "queue_next": "followup-generation",
}

NODE_EXPECTED_OUTPUTS = {
    "plan": ["plan_summary", "handoff_ready"],
    "prepare": ["environment_report", "adapter_resolution"],
    "execute": ["stage_statuses", "execution_summary"],
    "monitor": ["monitor_report", "artifact_readiness"],
    "analyze": ["analysis", "next_action"],
    "diagnose": ["diagnosis"],
    "patch": ["patch_request", "patch_actions"],
    "propose_next": ["next_manifest"],
    "queue_next": ["queued_followup"],
}

TERMINAL_NODE_ORDER = [
    "plan",
    "prepare",
    "execute",
    "monitor",
    "analyze",
    "diagnose",
    "patch",
    "propose_next",
    "queue_next",
]

SUCCESS_RESULT_STATUSES = {"completed", "success", "accepted", "ok"}
FAILURE_RESULT_STATUSES = {"failed", "error"}
REJECTED_RESULT_STATUSES = {"rejected"}
SUPERSEDED_RESULT_STATUSES = {"superseded"}


def build_dispatch_plan(state: Dict[str, Any], manifest: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    packets = []
    for node in state["controller_graph"]:
        packets.append(
            {
                "node": node["node"],
                "agent_id": node["agent"],
                "native_role": node.get("native_role"),
                "skill_name": node.get("skill_name"),
                "dispatch_status": node.get("dispatch_status"),
                "packet_id": node.get("packet_id"),
                "depends_on": node_dependencies(node["node"], state),
            }
        )
    return {
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "queue_owner": state["queue_owner"],
        "dispatch_mode": state["dispatch_mode"],
        "authority_source": state["authority_source"],
        "native_harness": registry.get("native_harness", {}),
        "packets": packets,
        "notes": [
            "Codex(default) is the sole queue owner.",
            "Packets are executed by native Codex roles and ingested back into controller state.",
        ],
    }


def node_dependencies(node_name: str, state: Dict[str, Any]) -> List[str]:
    order = [entry["node"] for entry in state["controller_graph"]]
    idx = order.index(node_name)
    if idx == 0:
        return []
    previous = order[idx - 1]
    return [previous]


def enrich_controller_graph(state: Dict[str, Any], registry: Dict[str, Any]) -> None:
    agent_map = {agent["agent_id"]: agent for agent in registry.get("agents", [])}
    for node in state["controller_graph"]:
        agent = agent_map[node["agent"]]
        node.setdefault("native_role", agent.get("native_role") or agent.get("fallback_native_role") or agent["role"])
        node.setdefault("skill_name", resolve_skill_name(node["node"], agent))
        node.setdefault("packet_id", packet_id(state["run_id"], node["node"]))
        node.setdefault("dispatch_status", "blocked")
        node.setdefault("result_path", str((Path(state["run_dir"]) / "agent_results" / f"{node['node']}.json").resolve()))
        node.setdefault("packet_path", str((Path(state["run_dir"]) / "agent_packets" / f"{node['node']}.json").resolve()))


def resolve_skill_name(node_name: str, agent: Dict[str, Any]) -> str:
    if node_name in NODE_SKILLS:
        return NODE_SKILLS[node_name]
    preferred = agent.get("preferred_skills", [])
    return preferred[0] if preferred else "controller-decomposition"


def packet_id(run_id: str, node_name: str) -> str:
    return f"{run_id}:{node_name}"


def packet_payload(
    *,
    state: Dict[str, Any],
    manifest: Dict[str, Any],
    node: Dict[str, Any],
    registry: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "packet_id": node["packet_id"],
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "controller_node": node["node"],
        "controller_agent": node["agent"],
        "native_role": node["native_role"],
        "required_skill": node["skill_name"],
        "queue_owner": state["queue_owner"],
        "inputs": build_packet_inputs(state, manifest, node),
        "expected_outputs": NODE_EXPECTED_OUTPUTS.get(node["node"], []),
        "acceptance_checks": acceptance_checks(state, manifest, node),
        "allowed_mutation_paths": allowed_mutation_paths(manifest, node["node"]),
        "result_ingest_target": node["result_path"],
        "harness": registry.get("native_harness", {}),
    }


def build_packet_inputs(state: Dict[str, Any], manifest: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "hypothesis": state.get("hypothesis"),
        "baseline_refs": state.get("baseline_refs", []),
        "run_dir": state["run_dir"],
        "workspace_root": state["workspace_root"],
        "manifest_path": state["manifest_path"],
        "active_stage_statuses": {stage["name"]: stage["status"] for stage in state.get("stages", [])},
    }
    if node["node"] == "execute":
        payload["stages"] = [stage for stage in state.get("stages", []) if stage["status"] in {"planned", "failed"}]
    if node["node"] == "analyze":
        payload["acceptance_criteria"] = manifest.get("acceptance_criteria", {})
    if node["node"] == "patch":
        payload["allowed_paths"] = manifest.get("patch_policy", {}).get("allowed_paths", [])
        payload["diagnosis_ref"] = state.get("diagnosis_ref")
    if node["node"] in {"propose_next", "queue_next"}:
        payload["next_step_policy"] = manifest.get("next_step_policy", {})
        payload["analysis_ref"] = state.get("analysis_ref")
    return payload


def acceptance_checks(state: Dict[str, Any], manifest: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {"dispatch_status_must_be": "reported"}
    if node["node"] == "execute":
        checks["required_stage_status"] = manifest.get("acceptance_criteria", {}).get("required_stage_status", {})
    elif node["node"] == "analyze":
        checks["must_produce"] = ["result_classification", "next_action"]
    elif node["node"] == "diagnose":
        checks["must_produce"] = ["summary", "suspected_causes"]
    elif node["node"] == "patch":
        checks["must_produce"] = ["summary", "change_intent"]
    elif node["node"] == "propose_next":
        checks["must_produce"] = ["experiment_id", "hypothesis"]
    elif node["node"] == "queue_next":
        checks["must_produce"] = ["queued_followup"]
    return checks


def allowed_mutation_paths(manifest: Dict[str, Any], node_name: str) -> List[str]:
    if node_name == "patch":
        return manifest.get("patch_policy", {}).get("allowed_paths", [])
    if node_name in {"propose_next", "queue_next"}:
        return ["experiments/manifests/", "knowledge/"]
    return []


def classify_result_status(result: Dict[str, Any]) -> str:
    status = str(result.get("status", "")).lower()
    if status in SUCCESS_RESULT_STATUSES:
        return "accepted"
    if status in FAILURE_RESULT_STATUSES:
        return "accepted"
    if status in REJECTED_RESULT_STATUSES:
        return "rejected"
    if status in SUPERSEDED_RESULT_STATUSES:
        return "superseded"
    return "accepted"


def graph_node_by_name(state: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    for node in state["controller_graph"]:
        if node["node"] == node_name:
            return node
    raise KeyError(node_name)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

AGENT_REGISTRY_PATH = Path("agents/registry.json")
DEFAULT_GRAPH = [
    ("plan", "planner"),
    ("prepare", "executor"),
    ("execute", "executor"),
    ("monitor", "monitor"),
    ("analyze", "analyst"),
    ("diagnose", "analyst"),
    ("patch", "maintainer"),
    ("propose_next", "planner"),
    ("queue_next", "codex"),
]


class AgentRegistryError(ValueError):
    """Raised when the agent registry is malformed."""


def load_agent_registry(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / AGENT_REGISTRY_PATH
    if not path.exists():
        raise AgentRegistryError(f"Agent registry does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_agent_registry(payload)
    return payload


def validate_agent_registry(payload: Dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AgentRegistryError("Only agent registry schema_version=1 is supported")
    agents = payload.get("agents")
    if not isinstance(agents, list) or not agents:
        raise AgentRegistryError("Agent registry must define a non-empty 'agents' array")
    ids = set()
    for agent in agents:
        agent_id = agent.get("agent_id")
        if not agent_id or not isinstance(agent_id, str):
            raise AgentRegistryError("Each agent must have a non-empty string agent_id")
        if agent_id in ids:
            raise AgentRegistryError(f"Duplicate agent id: {agent_id}")
        native_role = agent.get("native_role")
        if native_role is not None and not isinstance(native_role, str):
            raise AgentRegistryError(f"native_role must be a string when present for agent_id={agent_id}")
        ids.add(agent_id)
    controller = payload.get("top_level_controller")
    if controller not in ids:
        raise AgentRegistryError("top_level_controller must refer to a known agent_id")


def bootstrap_agent_state(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    state: Dict[str, Dict[str, Any]] = {}
    for agent in registry["agents"]:
        state[agent["agent_id"]] = {
            "role": agent["role"],
            "native_role": agent.get("native_role"),
            "fallback_native_role": agent.get("fallback_native_role"),
            "status": "idle",
            "current_node": None,
            "last_handoff_at": None,
            "last_output": None,
            "policy_scope": agent.get("policy_scope", []),
            "outputs": agent.get("outputs", []),
            "preferred_skills": agent.get("preferred_skills", []),
            "handoff_to": agent.get("handoff_to", []),
        }
    return state


def build_controller_graph(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    graph = manifest.get("controller_graph")
    if isinstance(graph, list) and graph:
        return [
            {
                "node": node["node"],
                "agent": node["agent"],
                "status": "planned",
                "started_at": None,
                "finished_at": None,
                "notes": [],
            }
            for node in graph
        ]
    return [
        {
            "node": node,
            "agent": agent,
            "status": "planned",
            "started_at": None,
            "finished_at": None,
            "notes": [],
        }
        for node, agent in DEFAULT_GRAPH
    ]


def set_active_agent(state: Dict[str, Any], agent_id: str, *, node: str | None = None) -> None:
    state["active_agent"] = agent_id
    for current_id, agent_state in state["agents"].items():
        if current_id == agent_id:
            agent_state["status"] = "active"
            agent_state["current_node"] = node
        elif agent_state["status"] == "active":
            agent_state["status"] = "idle"
            agent_state["current_node"] = None


def update_graph_node(state: Dict[str, Any], node_name: str, *, status: str, timestamp: str, note: str | None = None) -> Dict[str, Any]:
    for node in state["controller_graph"]:
        if node["node"] == node_name:
            node["status"] = status
            if status == "running":
                node["started_at"] = timestamp
            if status in {"completed", "failed", "skipped"}:
                node["finished_at"] = timestamp
            if note:
                node["notes"].append(note)
            return node
    raise AgentRegistryError(f"Unknown controller node: {node_name}")

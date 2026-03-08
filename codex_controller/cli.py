from __future__ import annotations

import argparse
import json

from .manifest import describe_manifest, load_manifest
from .runtime import (
    advance_run,
    emit_packets,
    finalize_run,
    ingest_agent_result,
    prepare_run,
    resume_run,
    run_manifest,
    summarize_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex-first experiment controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a manifest")
    validate_parser.add_argument("manifest", help="Path to manifest JSON")

    describe_parser = subparsers.add_parser("describe", help="Describe a manifest")
    describe_parser.add_argument("manifest", help="Path to manifest JSON")

    prepare_parser = subparsers.add_parser("prepare-run", help="Create a packet-driven native multi-agent run")
    prepare_parser.add_argument("manifest", help="Path to manifest JSON")
    prepare_parser.add_argument("--run-id", help="Explicit run id")
    prepare_parser.add_argument("--from-stage", help="Start from a given stage name")
    prepare_parser.add_argument("--through-stage", help="Stop after a given stage name")

    emit_parser = subparsers.add_parser("emit-packets", help="Emit ready dispatch packets for a prepared run")
    emit_parser.add_argument("run_dir", help="Path to controller run directory")
    emit_parser.add_argument("--node", action="append", dest="nodes", help="Emit only the specified controller node")

    ingest_parser = subparsers.add_parser("ingest-agent-result", help="Ingest a native agent result into controller state")
    ingest_parser.add_argument("run_dir", help="Path to controller run directory")
    ingest_parser.add_argument("node", help="Controller node name")
    ingest_parser.add_argument("result", help="Path to agent result JSON")

    advance_parser = subparsers.add_parser("advance-run", help="Accept reported packet results and advance controller state")
    advance_parser.add_argument("run_dir", help="Path to controller run directory")

    finalize_parser = subparsers.add_parser("finalize-run", help="Finalize a packet-driven run after all packets are resolved")
    finalize_parser.add_argument("run_dir", help="Path to controller run directory")

    run_parser = subparsers.add_parser("run", help="Compatibility path: create and execute a controller-managed run")
    run_parser.add_argument("manifest", help="Path to manifest JSON")
    run_parser.add_argument("--run-id", help="Explicit run id")
    run_parser.add_argument("--dry-run", action="store_true", help="Create state and resolve commands without executing")
    run_parser.add_argument("--from-stage", help="Start from a given stage name")
    run_parser.add_argument("--through-stage", help="Stop after a given stage name")

    resume_parser = subparsers.add_parser("resume", help="Compatibility path: resume an existing controller run")
    resume_parser.add_argument("run_dir", help="Path to controller run directory")
    resume_parser.add_argument("--dry-run", action="store_true", help="Resolve resumable stages without executing")
    resume_parser.add_argument("--through-stage", help="Stop after a given stage name")

    status_parser = subparsers.add_parser("status", help="Summarize a controller run")
    status_parser.add_argument("run_dir", help="Path to controller run directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        manifest_path, manifest, repo_root = load_manifest(args.manifest)
        print(json.dumps({
            "status": "ok",
            "manifest": str(manifest_path),
            "experiment_id": manifest["experiment_id"],
            "repo_root": str(repo_root),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "describe":
        _, manifest, _ = load_manifest(args.manifest)
        print(describe_manifest(manifest))
        return 0

    if args.command == "prepare-run":
        run_dir = prepare_run(
            args.manifest,
            run_id=args.run_id,
            from_stage=args.from_stage,
            through_stage=args.through_stage,
        )
        print(run_dir)
        return 0

    if args.command == "emit-packets":
        packets = emit_packets(args.run_dir, nodes=args.nodes)
        print(json.dumps([str(path) for path in packets], ensure_ascii=False, indent=2))
        return 0

    if args.command == "ingest-agent-result":
        result_path = ingest_agent_result(args.run_dir, node_name=args.node, result_path_arg=args.result)
        print(result_path)
        return 0

    if args.command == "advance-run":
        run_dir = advance_run(args.run_dir)
        print(run_dir)
        return 0

    if args.command == "finalize-run":
        run_dir = finalize_run(args.run_dir)
        print(run_dir)
        return 0

    if args.command == "run":
        run_dir = run_manifest(
            args.manifest,
            run_id=args.run_id,
            dry_run=args.dry_run,
            from_stage=args.from_stage,
            through_stage=args.through_stage,
        )
        print(run_dir)
        return 0

    if args.command == "resume":
        run_dir = resume_run(args.run_dir, dry_run=args.dry_run, through_stage=args.through_stage)
        print(run_dir)
        return 0

    if args.command == "status":
        print(json.dumps(summarize_run(args.run_dir), ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2

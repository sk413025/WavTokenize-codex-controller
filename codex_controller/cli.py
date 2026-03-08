from __future__ import annotations

import argparse
import json

from .manifest import describe_manifest, load_manifest
from .monitor import inspect_run, write_monitor_result
from .runtime import resume_run, run_manifest, summarize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin run helper for official experiment manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a manifest")
    validate_parser.add_argument("manifest", help="Path to manifest JSON")

    describe_parser = subparsers.add_parser("describe", help="Describe a manifest")
    describe_parser.add_argument("manifest", help="Path to manifest JSON")

    monitor_parser = subparsers.add_parser("monitor-run", help="Inspect stage logs and artifacts and write a monitor report")
    monitor_parser.add_argument("run_dir", help="Path to a persisted run directory")
    monitor_parser.add_argument("--result-path", help="Explicit path for the monitor report JSON")
    monitor_parser.add_argument("--stall-seconds", type=int, default=1800, help="Mark a running stage as stalled after this many seconds without log updates")
    monitor_parser.add_argument("--print-report", action="store_true", help="Print the monitor report JSON to stdout")

    run_parser = subparsers.add_parser("run", help="Execute an official manifest and persist run artifacts")
    run_parser.add_argument("manifest", help="Path to manifest JSON")
    run_parser.add_argument("--run-id", help="Explicit run id")
    run_parser.add_argument("--dry-run", action="store_true", help="Create state and resolve commands without executing")
    run_parser.add_argument("--from-stage", help="Start from a given stage name")
    run_parser.add_argument("--through-stage", help="Stop after a given stage name")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing persisted run")
    resume_parser.add_argument("run_dir", help="Path to a persisted run directory")
    resume_parser.add_argument("--dry-run", action="store_true", help="Resolve resumable stages without executing")
    resume_parser.add_argument("--through-stage", help="Stop after a given stage name")

    status_parser = subparsers.add_parser("status", help="Summarize a persisted run")
    status_parser.add_argument("run_dir", help="Path to a persisted run directory")

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

    if args.command == "monitor-run":
        result_path = write_monitor_result(
            args.run_dir,
            result_path_arg=args.result_path,
            stall_seconds=args.stall_seconds,
        )
        if args.print_report:
            print(json.dumps(inspect_run(args.run_dir, stall_seconds=args.stall_seconds), ensure_ascii=False, indent=2))
        else:
            print(result_path)
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

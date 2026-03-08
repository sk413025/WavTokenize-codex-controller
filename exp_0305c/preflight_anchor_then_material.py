from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE

DEFAULT_TRAIN_CACHE = "/home/sbplab/ruizi/WavTokenize-feature-analysis/data/train_cache_filtered.pt"
DEFAULT_VAL_CACHE = "/home/sbplab/ruizi/WavTokenize-feature-analysis/data/val_cache_filtered.pt"
DEFAULT_MATERIAL_ENCODER_CKPT = "exp_0227/runs/enc_mrd_fm_epoch_20260227_024953/best_model_val_total.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for exp_0305c -> exp_0304 sequential execution")
    parser.add_argument("--primary_gpu", default="cuda:0")
    parser.add_argument("--secondary_gpu", default="cuda:1")
    parser.add_argument("--wavtok_root", default="/home/sbplab/ruizi/WavTokenizer-main")
    parser.add_argument("--wavtok_config", default=WAVTOK_CONFIG)
    parser.add_argument("--wavtok_ckpt", default=WAVTOK_CKPT)
    parser.add_argument("--train_cache", default=DEFAULT_TRAIN_CACHE or str(TRAIN_CACHE))
    parser.add_argument("--val_cache", default=DEFAULT_VAL_CACHE or str(VAL_CACHE))
    parser.add_argument("--material_encoder_ckpt", default=DEFAULT_MATERIAL_ENCODER_CKPT)
    parser.add_argument("--smoke_min_free_mib", type=int, default=1800)
    parser.add_argument("--real_min_free_mib", type=int, default=5500)
    parser.add_argument("--report_path", required=True)
    return parser.parse_args()


def gpu_index(gpu: str) -> int | None:
    if gpu.startswith("cuda:"):
        gpu = gpu.split(":", 1)[1]
    try:
        return int(gpu)
    except ValueError:
        return None


def query_all_gpus() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        line_index, name, total_mib, used_mib = parts
        total = int(total_mib)
        used = int(used_mib)
        records.append({
            "status": "ok",
            "index": int(line_index),
            "name": name,
            "memory_total_mib": total,
            "memory_used_mib": used,
            "memory_free_mib": total - used,
        })
    return records


def query_gpu(selector: str) -> dict[str, Any]:
    index = gpu_index(selector)
    if index is None:
        return {"status": "unresolved", "summary": f"GPU selector {selector!r} is not a concrete cuda index."}
    try:
        for record in query_all_gpus():
            if record["index"] == index:
                return record
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"status": "unavailable", "summary": f"nvidia-smi unavailable: {exc}"}
    return {"status": "missing", "summary": f"GPU index {index} not found in nvidia-smi output."}


def asset_record(label: str, path_str: str, *, required: bool) -> dict[str, Any]:
    path = Path(path_str)
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "required": required,
    }


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        asset_record("wavtok_root", args.wavtok_root, required=True),
        asset_record("wavtok_config", args.wavtok_config, required=True),
        asset_record("wavtok_ckpt", args.wavtok_ckpt, required=True),
        asset_record("train_cache", args.train_cache, required=True),
        asset_record("val_cache", args.val_cache, required=True),
        asset_record("material_encoder_ckpt", args.material_encoder_ckpt, required=False),
    ]
    missing_required = [record for record in records if record["required"] and not record["exists"]]
    missing_optional = [record for record in records if not record["required"] and not record["exists"]]

    primary = query_gpu(args.primary_gpu)
    secondary = query_gpu(args.secondary_gpu)
    fallback_mode = "material_encoder_ckpt" if not missing_optional else "pretrained_material_stage"

    if missing_required:
        status = "blocked"
        recommended_next_action = "bind_missing_assets"
        summary = "Required assets are missing for the exp_0305c -> exp_0304 chain."
    elif primary.get("status") != "ok" or secondary.get("status") != "ok":
        status = "blocked"
        recommended_next_action = "fix_gpu_probe"
        summary = "GPU readiness could not be established for one or both target devices."
    else:
        primary_free = int(primary["memory_free_mib"])
        secondary_free = int(secondary["memory_free_mib"])
        if min(primary_free, secondary_free) >= args.real_min_free_mib:
            status = "ready_for_real_run"
            recommended_next_action = "launch_short_or_full_chain"
            summary = "Both sequential GPUs have enough headroom for a real chain launch."
        elif min(primary_free, secondary_free) >= args.smoke_min_free_mib:
            status = "ready_for_smoke"
            recommended_next_action = "launch_smoke_gate"
            summary = "Both sequential GPUs can support a smoke validation chain."
        else:
            status = "blocked"
            recommended_next_action = "wait_for_gpu_capacity"
            summary = "One or both sequential GPUs are below the smoke threshold."

    report = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "exp0305c_to_exp0304_controller",
        "asset_bindings": records,
        "missing_assets": [record["label"] for record in missing_required],
        "fallback_mode": fallback_mode,
        "gpu_readiness": {
            "primary_gpu": primary,
            "secondary_gpu": secondary,
        },
        "recommended_next_action": recommended_next_action,
        "summary": summary,
        "thresholds": {
            "smoke_min_free_mib": args.smoke_min_free_mib,
            "real_min_free_mib": args.real_min_free_mib,
        },
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("exp_0305c -> exp_0304 preflight")
    print(f"Report: {report_path}")
    print(f"Fallback mode: {fallback_mode}")
    if primary.get("status") == "ok":
        print(f"Primary GPU {primary['index']} {primary['name']}: free {primary['memory_free_mib']} MiB")
    else:
        print(f"Primary GPU probe: {primary.get('summary', primary['status'])}")
    if secondary.get("status") == "ok":
        print(f"Secondary GPU {secondary['index']} {secondary['name']}: free {secondary['memory_free_mib']} MiB")
    else:
        print(f"Secondary GPU probe: {secondary.get('summary', secondary['status'])}")
    print(f"PREFLIGHT STATUS: {status}")
    print(f"NEXT ACTION: {recommended_next_action}")
    return 0 if status in {"ready_for_smoke", "ready_for_real_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

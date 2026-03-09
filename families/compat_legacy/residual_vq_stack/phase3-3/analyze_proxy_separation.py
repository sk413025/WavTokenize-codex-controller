"""
Phase 3-3 Exp7d: Proxy separation analysis for split children codes.

Goal:
  - Validate H1: a hot code mixes multiple modes (speech/noise), and split children
    correlate with a noise proxy (default: sample-level SNR).

This script:
  1) Loads a Phase3-3 run dir (expects `final_model.pt` and `split_history.json`)
  2) Runs inference on the validation set to obtain layer0 codes
  3) Computes per-code proxy stats and per-split-group child separation metrics
  4) Writes artifacts into a timestamped output directory (incl. copied split_history.json)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_3d_audio(x: torch.Tensor) -> torch.Tensor:
    # Expected by WavTokenizer encoder: [B, 1, T]
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x


def _estimate_snr_db(
    noisy_audio: torch.Tensor,
    clean_audio: torch.Tensor,
    lengths: torch.Tensor | None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Estimate sample-level SNR in dB.

    SNR = 10 * log10(P_signal / P_noise)
      P_signal = mean(clean^2)
      P_noise  = mean((noisy-clean)^2)
    """
    # Accept [B, T] or [B, 1, T]
    if noisy_audio.dim() == 3:
        noisy_audio = noisy_audio.squeeze(1)
    if clean_audio.dim() == 3:
        clean_audio = clean_audio.squeeze(1)

    if noisy_audio.shape != clean_audio.shape:
        min_len = min(noisy_audio.shape[-1], clean_audio.shape[-1])
        noisy_audio = noisy_audio[..., :min_len]
        clean_audio = clean_audio[..., :min_len]

    B, T = noisy_audio.shape
    if lengths is None:
        valid = torch.ones(B, T, device=noisy_audio.device, dtype=noisy_audio.dtype)
    else:
        lengths = lengths.to(device=noisy_audio.device)
        idx = torch.arange(T, device=noisy_audio.device).unsqueeze(0)
        valid = (idx < lengths.unsqueeze(1)).to(noisy_audio.dtype)

    clean2 = (clean_audio * clean_audio) * valid
    noise = (noisy_audio - clean_audio)
    noise2 = (noise * noise) * valid

    denom = valid.sum(dim=1).clamp(min=1.0)
    p_signal = clean2.sum(dim=1) / denom
    p_noise = noise2.sum(dim=1) / denom

    snr = 10.0 * torch.log10((p_signal + eps) / (p_noise + eps))
    return snr


def _audio_lengths_to_frame_lengths(lengths: torch.Tensor | None, n_frames: int, hop: int = 320) -> torch.Tensor | None:
    if lengths is None:
        return None
    # ceil(length/hop)
    frame_lens = (lengths + hop - 1) // hop
    frame_lens = torch.clamp(frame_lens, min=0, max=int(n_frames))
    return frame_lens


def _init_device(requested: str) -> torch.device:
    """
    Best-effort device selection.
    - If requested is CUDA but CUDA init is flaky, fall back to CPU (and log it).
    """
    dev = torch.device(requested)
    if dev.type != "cuda":
        return dev

    # Fast availability check (can still be flaky); follow with a tiny alloc.
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        _ = torch.tensor([0.0], device=dev)
        torch.cuda.synchronize(dev)
        return dev
    except Exception:
        return torch.device("cpu")


@dataclass
class CodeStats:
    n_tokens: int
    mean: float | None
    std: float | None


def _stats_from_sums(n: int, s1: float, s2: float) -> CodeStats:
    if n <= 0:
        return CodeStats(n_tokens=0, mean=None, std=None)
    mean = s1 / float(n)
    var = max(0.0, (s2 / float(n)) - (mean * mean))
    return CodeStats(n_tokens=int(n), mean=float(mean), std=float(math.sqrt(var)))


def _effect_size_d(
    a: CodeStats,
    b: CodeStats,
) -> dict[str, float | None]:
    """
    Weighted Cohen's d using population variances (streaming-friendly).
    d = (mean_a - mean_b) / sqrt(pooled_var)
    pooled_var = (n_a*var_a + n_b*var_b)/(n_a+n_b)
    """
    if a.n_tokens <= 1 or b.n_tokens <= 1:
        return {"d": None, "abs_d": None, "mean_diff": None, "pooled_std": None}
    if a.mean is None or b.mean is None or a.std is None or b.std is None:
        return {"d": None, "abs_d": None, "mean_diff": None, "pooled_std": None}

    var_a = float(a.std) ** 2
    var_b = float(b.std) ** 2
    pooled_var = (a.n_tokens * var_a + b.n_tokens * var_b) / float(a.n_tokens + b.n_tokens)
    pooled_std = math.sqrt(max(pooled_var, 0.0))
    if pooled_std < 1e-12:
        return {"d": None, "abs_d": None, "mean_diff": float(a.mean - b.mean), "pooled_std": float(pooled_std)}
    d = (float(a.mean) - float(b.mean)) / pooled_std
    return {"d": float(d), "abs_d": float(abs(d)), "mean_diff": float(a.mean - b.mean), "pooled_std": float(pooled_std)}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp7d proxy separation analysis (SNR) for split children codes")
    parser.add_argument("--base_run_dir", type=str, required=True, help="Phase3-3 run dir containing final_model.pt + split_history.json")
    parser.add_argument("--output_dir", type=str, default="", help="Output dir (default: exp_0128/phase3-3/run_exp7d_proxy_..._<ts>)")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for inference (fallback to CPU if CUDA init fails)")
    parser.add_argument("--batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Validation dataloader workers")
    parser.add_argument("--max_batches", type=int, default=50, help="Max val batches to process")
    parser.add_argument("--min_child_tokens", type=int, default=50, help="Min tokens per child for a group to be considered for P3")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    base_run_dir = Path(args.base_run_dir)
    if not base_run_dir.exists():
        raise FileNotFoundError(f"base_run_dir not found: {base_run_dir}")

    base_split_path = base_run_dir / "split_history.json"
    base_ckpt_path = base_run_dir / "final_model.pt"
    if not base_split_path.exists():
        raise FileNotFoundError(f"missing split_history.json: {base_split_path}")
    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"missing final_model.pt: {base_ckpt_path}")

    # Import order is important: models_rvq injects self-supervised repo into sys.path,
    # which exp_1226.data_curriculum depends on.
    from exp_0128.phase3.residual_vq.models_rvq import TeacherStudentRVQ
    from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
    from exp_1226.data_curriculum import CurriculumDataset, collate_fn_curriculum

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = repo_root / "exp_0128" / "phase3-3" / f"run_exp7d_proxy_snr_from_{base_run_dir.name}_{_now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "analysis.log"
    cfg_path = out_dir / "config.json"
    out_json_path = out_dir / "proxy_separation.json"

    def _log(msg: str) -> None:
        line = f"[{_iso_now()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    _log("=== Exp7d proxy separation (SNR) ===")
    _log(f"base_run_dir: {base_run_dir}")
    _log(f"output_dir: {out_dir}")
    _log(f"requested_device: {args.device}")

    with open(base_split_path, "r") as f:
        split_history: list[dict[str, Any]] = json.load(f)

    # Persist split_history.json in this Exp7d output dir (requirement: each run has it).
    with open(out_dir / "split_history.json", "w") as f:
        json.dump(split_history, f, indent=2)

    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})

    device = _init_device(args.device)
    _log(f"selected_device: {device}")

    # Build model (match training config)
    model = TeacherStudentRVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=int(ckpt_cfg.get("lora_rank", 256)),
        lora_alpha=int(ckpt_cfg.get("lora_alpha", 512)),
        intermediate_indices=[3, 6],
        device=device,
        n_rvq_layers=int(ckpt_cfg.get("n_rvq_layers", 4)),
        rvq_codebook_size=int(ckpt_cfg.get("rvq_codebook_size", 2048)),
        rvq_update=str(ckpt_cfg.get("rvq_update", "ema")),
        ema_decay=float(ckpt_cfg.get("ema_decay", 0.99)),
        ema_eps=float(ckpt_cfg.get("ema_eps", 1e-5)),
        ema_dead_code_threshold=int(ckpt_cfg.get("ema_dead_code_threshold", 0)),
        ema_usage_penalty=float(ckpt_cfg.get("ema_usage_penalty", 0.0)),
    )

    # Load checkpoint state
    model.student.load_state_dict(ckpt["student_state_dict"], strict=True)
    model.rvq.load_state_dict(ckpt["rvq_state_dict"], strict=True)
    model.eval()

    K = int(model.rvq_codebook_size)
    counts = torch.zeros(K, dtype=torch.long)
    sum_snr = torch.zeros(K, dtype=torch.float64)
    sum_snr2 = torch.zeros(K, dtype=torch.float64)

    # Dataloader (val only). Avoid loading the (large) train cache to reduce RAM.
    _log(f"VAL_CACHE: {VAL_CACHE}")
    val_dataset = CurriculumDataset(
        VAL_CACHE,
        filter_clean_to_clean=True,
        compute_snr=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn_curriculum,
        pin_memory=False,
    )

    max_batches = int(args.max_batches)
    min_child_tokens = int(args.min_child_tokens)

    _log(f"max_batches: {max_batches}")
    _log(f"min_child_tokens: {min_child_tokens}")

    # Inference loop (student-only; skip teacher to speed up).
    hop = 320
    started = time.time()
    n_seen_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            lengths = batch.get("lengths", None)
            if lengths is not None:
                lengths_cpu = lengths.cpu()
            else:
                lengths_cpu = None

            noisy = _ensure_3d_audio(batch["noisy_audio"]).to(device)
            clean = _ensure_3d_audio(batch["clean_audio"]).to(device)

            # Proxy: sample-level SNR (assign to all frames of the sample)
            snr_db = _estimate_snr_db(noisy, clean, lengths_cpu).detach().cpu()  # [B]

            # Codes: layer0 codes [B, T_frames]
            student_encoder_out, _ = model.student_extractor(noisy)
            vq_out = model.rvq(student_encoder_out, frame_rate=75, bandwidth=0.075)
            all_layer_codes = vq_out["all_layer_codes"]  # [n_layers, B, T]
            layer0 = all_layer_codes[0].detach().cpu()  # [B, T]

            B, T_frames = layer0.shape
            frame_lens = _audio_lengths_to_frame_lengths(lengths_cpu, n_frames=int(T_frames), hop=hop)

            for b in range(B):
                L = int(frame_lens[b].item()) if frame_lens is not None else int(T_frames)
                if L <= 0:
                    continue
                codes = layer0[b, :L].to(torch.long)
                cnt = torch.bincount(codes, minlength=K)  # [K], long
                snr = float(snr_db[b].item())
                counts += cnt
                cnt_f = cnt.to(torch.float64)
                sum_snr += cnt_f * snr
                sum_snr2 += cnt_f * (snr * snr)

            n_seen_batches += 1
            if (i + 1) % 10 == 0:
                _log(f"progress: {i + 1}/{max_batches} batches")

    elapsed = time.time() - started
    _log(f"done: processed_batches={n_seen_batches}, elapsed_s={elapsed:.1f}")

    # Prepare results per split group
    groups_out: list[dict[str, Any]] = []
    pass_groups = 0
    abs_ds: list[float] = []

    for ev in split_history:
        parent = int(ev["parent"])
        children = [int(x) for x in ev["children"]]
        if len(children) != 2:
            _log(f"skip group with unexpected children count: parent={parent}, children={children}")
            continue

        p = _stats_from_sums(int(counts[parent].item()), float(sum_snr[parent].item()), float(sum_snr2[parent].item()))
        c0 = _stats_from_sums(
            int(counts[children[0]].item()),
            float(sum_snr[children[0]].item()),
            float(sum_snr2[children[0]].item()),
        )
        c1 = _stats_from_sums(
            int(counts[children[1]].item()),
            float(sum_snr[children[1]].item()),
            float(sum_snr2[children[1]].item()),
        )
        eff = _effect_size_d(c0, c1)

        group_pass = False
        if eff["abs_d"] is not None:
            if c0.n_tokens >= min_child_tokens and c1.n_tokens >= min_child_tokens and float(eff["abs_d"]) >= 0.5:
                group_pass = True

        if group_pass:
            pass_groups += 1
        if eff["abs_d"] is not None:
            abs_ds.append(float(eff["abs_d"]))

        groups_out.append(
            {
                "step": int(ev.get("step", -1)),
                "layer": int(ev.get("layer", 0)),
                "parent": parent,
                "children": children,
                "proxy": "snr_db",
                "stats": {
                    "parent": asdict(p),
                    "child0": asdict(c0),
                    "child1": asdict(c1),
                },
                "child_separation": eff,
                "group_pass_effect_size_0p5": bool(group_pass),
            }
        )

    abs_d_median = _median(abs_ds)
    abs_d_mean = float(sum(abs_ds) / len(abs_ds)) if abs_ds else None

    # Overall P3: require at least half of the groups (rounded down to >=1) to pass.
    total_groups = len(groups_out)
    required = max(1, total_groups // 2) if total_groups > 0 else 1
    p3_pass = bool(pass_groups >= required)

    result = {
        "timestamp": _iso_now(),
        "base_run_dir": str(base_run_dir),
        "base_checkpoint": str(base_ckpt_path),
        "device": str(device),
        "max_batches": max_batches,
        "min_child_tokens": min_child_tokens,
        "proxy": "snr_db",
        "summary": {
            "groups_total": total_groups,
            "groups_pass": int(pass_groups),
            "required_for_p3": int(required),
            "abs_d_median": abs_d_median,
            "abs_d_mean": abs_d_mean,
            "p3_pass": bool(p3_pass),
        },
        "groups": groups_out,
    }

    with open(cfg_path, "w") as f:
        json.dump(
            {
                "timestamp": _iso_now(),
                "base_run_dir": str(base_run_dir),
                "base_split_history": str(base_split_path),
                "base_checkpoint": str(base_ckpt_path),
                "requested_device": str(args.device),
                "selected_device": str(device),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "max_batches": max_batches,
                "min_child_tokens": min_child_tokens,
            },
            f,
            indent=2,
        )

    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)

    _log(f"Wrote: {cfg_path}")
    _log(f"Wrote: {out_json_path}")
    _log(f"P3_pass={p3_pass} (pass_groups={pass_groups}/{total_groups}, required={required})")


if __name__ == "__main__":
    main()

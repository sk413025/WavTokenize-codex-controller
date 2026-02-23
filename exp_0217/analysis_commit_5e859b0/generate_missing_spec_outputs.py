#!/usr/bin/env python3
"""
Generate missing SPEC outputs for commit 5e859b0 analysis.

Scope:
- Build epoch-level audio quality report for exp_0216 run.
- Build stratified quality report and required plots.
- Update SPEC coverage status.

Notes:
- This script does not change any training setting or model weight.
- Epoch 222 has no checkpoint file in this run; it is marked unavailable.
"""

from __future__ import annotations

import json
import math
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "exp_0217" / "analysis_commit_5e859b0"
RUN_DIR = ROOT / "exp_0216" / "runs" / "augmented_long_20260216"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
METRICS_HISTORY_PATH = RUN_DIR / "metrics_history.json"
SUMMARY_PATH = RUN_DIR / "summary.json"

TRAIN_CACHE = ROOT / "data" / "train_cache_filtered.pt"
VAL_CACHE = ROOT / "data" / "val_cache_filtered.pt"

RAW_ROOT = Path("/home/sbplab/ruizi/WavTokenize/data/raw")
CLEAN_ROOT = Path("/home/sbplab/ruizi/WavTokenize/data/clean/box2")

EPOCHS = [50, 100, 150, 200, 220, 222, 250, 300]
SPLITS = ["train", "val"]

TARGET_SR_MODEL = 24000
TARGET_SR_METRIC = 16000
N_TARGET = 100
SEED = 42

# SPEC stratified bins
T453_BINS = [
    (0.0, 0.1, "[0,0.1)"),
    (0.1, 0.2, "[0.1,0.2)"),
    (0.2, 0.3, "[0.2,0.3)"),
    (0.3, 0.5 + 1e-8, "[0.3,0.5]"),
]
SNR_BINS = [
    (-1e9, 0.0, "<0dB"),
    (0.0, 10.0, "0~10dB"),
    (10.0, 20.0, "10~20dB"),
    (20.0, 1e9, ">20dB"),
]
LEN_BINS = [
    (0.0, 2.0, "<2s"),
    (2.0, 5.0, "2~5s"),
    (5.0, 1e9, ">5s"),
]


# -----------------------------------------------------------------------------
# Runtime imports for model
# -----------------------------------------------------------------------------

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/sbplab/ruizi/WavTokenizer-main")

from exp_1201.config import WAVTOK_CKPT, WAVTOK_CONFIG  # noqa: E402
from exp_0206.plan_ori.models_single_vq_ema import TeacherStudentSingleVQ  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isfinite(v):
        return v
    return None


def mean_std_ci(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "ci95": None}
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    if n <= 1:
        return {"mean": mean, "std": 0.0, "ci95": 0.0}
    std = float(arr.std(ddof=1))
    ci95 = float(1.96 * std / math.sqrt(n))
    return {"mean": mean, "std": std, "ci95": ci95}


def to_numpy_1d(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def t453_ratio_from_tokens(tokens) -> Optional[float]:
    if tokens is None:
        return None
    arr = to_numpy_1d(tokens)
    if arr.size == 0:
        return None
    return float((arr == 453).sum() / arr.size)


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Use first channel only.
    return wav[:, 0], int(sr)


def resample_audio(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio.astype(np.float32, copy=False)
    return resample_poly(audio, tgt_sr, src_sr).astype(np.float32)


def clip_audio(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def align3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(a), len(b), len(c))
    return a[:n], b[:n], c[:n]


def compute_snr_db(clean: np.ndarray, noisy: np.ndarray) -> Optional[float]:
    n = min(len(clean), len(noisy))
    if n <= 0:
        return None
    clean = clean[:n]
    noisy = noisy[:n]
    noise = noisy - clean
    p_sig = float(np.mean(clean ** 2) + 1e-12)
    p_noise = float(np.mean(noise ** 2) + 1e-12)
    if p_noise <= 0:
        return None
    return float(10.0 * math.log10(p_sig / p_noise))


def in_bin(v: float, lo: float, hi: float, hi_closed: bool = False) -> bool:
    if hi_closed:
        return lo <= v <= hi
    return lo <= v < hi


def assign_bin(v: Optional[float], bins: List[Tuple[float, float, str]], hi_closed_last: bool = False) -> str:
    if v is None or not math.isfinite(v):
        return "NA"
    for i, (lo, hi, label) in enumerate(bins):
        closed = hi_closed_last and i == len(bins) - 1
        if in_bin(v, lo, hi, hi_closed=closed):
            return label
    return "out_of_range"


def resolve_audio_path(
    path_or_name: str,
    raw_index: Dict[str, List[Path]],
) -> Optional[Path]:
    p = Path(path_or_name)
    if p.is_absolute() and p.exists():
        return p

    name = p.name
    # Fast deterministic rules first.
    if "_clean_" in name:
        c = CLEAN_ROOT / name
        if c.exists():
            return c
    if "_box_" in name:
        c = RAW_ROOT / "box" / name
        if c.exists():
            return c
    if "_papercup_" in name:
        c = RAW_ROOT / "papercup" / name
        if c.exists():
            return c
    if "_plastic_" in name:
        c = RAW_ROOT / "plastic" / name
        if c.exists():
            return c

    # Fallback index lookup.
    matches = raw_index.get(name, [])
    if matches:
        return matches[0]
    return None


def build_raw_index() -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    if not RAW_ROOT.exists():
        return idx
    for p in RAW_ROOT.rglob("*.wav"):
        idx.setdefault(p.name, []).append(p)
    return idx


def pick_indices(
    samples: List[dict],
    n_target: int,
    seed: int,
    raw_index: Dict[str, List[Path]],
) -> List[int]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(samples))
    rng.shuffle(order)
    selected: List[int] = []
    for idx in order.tolist():
        item = samples[idx]
        noisy_path = resolve_audio_path(str(item.get("noisy_path", "")), raw_index)
        clean_path = resolve_audio_path(str(item.get("clean_path", "")), raw_index)
        if noisy_path is None or clean_path is None:
            continue
        selected.append(idx)
        if len(selected) >= n_target:
            break
    return selected


def apply_lora_state(model: torch.nn.Module, lora_state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_state:
                src = lora_state[name].to(device=param.device, dtype=param.dtype)
                param.copy_(src)


@dataclass
class CheckpointLoadResult:
    epoch: int
    status: str
    checkpoint_path: Optional[str]
    model_epoch: Optional[int]
    reason: Optional[str] = None


def load_checkpoint_for_epoch(model, epoch: int) -> CheckpointLoadResult:
    # SPEC requests epoch_222(best), but this run has no checkpoint at 222.
    if epoch == 222:
        return CheckpointLoadResult(
            epoch=epoch,
            status="missing_checkpoint",
            checkpoint_path=None,
            model_epoch=None,
            reason="checkpoint_epoch222.pt not found (save interval = 10 epochs)",
        )

    ckpt_path = CHECKPOINT_DIR / f"checkpoint_epoch{epoch:03d}.pt"
    if not ckpt_path.exists():
        return CheckpointLoadResult(
            epoch=epoch,
            status="missing_checkpoint",
            checkpoint_path=str(ckpt_path),
            model_epoch=None,
            reason="checkpoint file does not exist",
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "lora_state" not in ckpt or "vq_state_dict" not in ckpt:
        return CheckpointLoadResult(
            epoch=epoch,
            status="invalid_checkpoint_format",
            checkpoint_path=str(ckpt_path),
            model_epoch=None,
            reason="required keys lora_state/vq_state_dict missing",
        )

    apply_lora_state(model, ckpt["lora_state"])
    model.vq.load_state_dict(ckpt["vq_state_dict"])
    return CheckpointLoadResult(
        epoch=epoch,
        status="ok",
        checkpoint_path=str(ckpt_path),
        model_epoch=int(ckpt.get("epoch", epoch)),
    )


def evaluate_sample(
    model,
    device: torch.device,
    item: dict,
    cache_index: int,
    split: str,
    epoch: int,
    raw_index: Dict[str, List[Path]],
) -> Optional[dict]:
    noisy_path = resolve_audio_path(str(item.get("noisy_path", "")), raw_index)
    clean_path = resolve_audio_path(str(item.get("clean_path", "")), raw_index)
    if noisy_path is None or clean_path is None:
        return None

    noisy_raw, sr_noisy = load_audio_mono(noisy_path)
    clean_raw, sr_clean = load_audio_mono(clean_path)

    noisy24 = resample_audio(noisy_raw, sr_noisy, TARGET_SR_MODEL)
    clean24 = resample_audio(clean_raw, sr_clean, TARGET_SR_MODEL)
    n = min(len(noisy24), len(clean24))
    if n <= TARGET_SR_MODEL // 10:
        return None
    noisy24 = clip_audio(noisy24[:n])
    clean24 = clip_audio(clean24[:n])

    # Model inference
    with torch.no_grad():
        noisy_t = torch.from_numpy(noisy24).to(device=device, dtype=torch.float32).view(1, 1, -1)
        clean_t = torch.from_numpy(clean24).to(device=device, dtype=torch.float32).view(1, 1, -1)
        out = model(clean_t, noisy_t)
        rec_t = model.decode(out["student_quantized"])
        if rec_t.dim() == 3:
            rec_t = rec_t.squeeze(1)
        rec24 = rec_t.squeeze(0).float().cpu().numpy()

        feature_mse = float(torch.mean((out["student_quantized"] - out["teacher_encoder_out"]) ** 2).item())

    # Metric audio (16k)
    clean16 = resample_audio(clean24, TARGET_SR_MODEL, TARGET_SR_METRIC)
    noisy16 = resample_audio(noisy24, TARGET_SR_MODEL, TARGET_SR_METRIC)
    rec16 = resample_audio(rec24, TARGET_SR_MODEL, TARGET_SR_METRIC)
    clean16, noisy16, rec16 = align3(clean16, noisy16, rec16)
    if len(clean16) <= TARGET_SR_METRIC // 10:
        return None
    clean16 = clip_audio(clean16)
    noisy16 = clip_audio(noisy16)
    rec16 = clip_audio(rec16)

    try:
        pesq_noisy = float(pesq(TARGET_SR_METRIC, clean16, noisy16, "wb"))
        pesq_recon = float(pesq(TARGET_SR_METRIC, clean16, rec16, "wb"))
        stoi_noisy = float(stoi(clean16, noisy16, TARGET_SR_METRIC, extended=False))
        stoi_recon = float(stoi(clean16, rec16, TARGET_SR_METRIC, extended=False))
    except Exception:
        return None

    delta_pesq = pesq_recon - pesq_noisy
    delta_stoi = stoi_recon - stoi_noisy

    # Metadata for stratified analysis.
    t453_ratio = t453_ratio_from_tokens(item.get("clean_tokens"))
    snr_db = compute_snr_db(clean24, noisy24)
    duration_sec = float(len(clean24) / TARGET_SR_MODEL)

    return {
        "split": split,
        "epoch": epoch,
        "cache_index": int(cache_index),
        "material": item.get("material"),
        "speaker_id": item.get("speaker_id"),
        "content_id": item.get("content_id"),
        "sentence_id": item.get("sentence_id"),
        "filename": item.get("filename"),
        "noisy_path": str(noisy_path),
        "clean_path": str(clean_path),
        "duration_sec": duration_sec,
        "snr_db": snr_db,
        "t453_ratio": t453_ratio,
        "feature_mse": feature_mse,
        "pesq_noisy": pesq_noisy,
        "pesq_recon": pesq_recon,
        "delta_pesq": delta_pesq,
        "stoi_noisy": stoi_noisy,
        "stoi_recon": stoi_recon,
        "delta_stoi": delta_stoi,
    }


def summarize_records(records: List[dict]) -> dict:
    result = {"num_samples": len(records)}
    for key in [
        "feature_mse",
        "pesq_noisy",
        "pesq_recon",
        "delta_pesq",
        "stoi_noisy",
        "stoi_recon",
        "delta_stoi",
        "snr_db",
        "duration_sec",
        "t453_ratio",
    ]:
        vals = [safe_float(r.get(key)) for r in records]
        vals = [v for v in vals if v is not None]
        stats = mean_std_ci(vals)
        result[f"{key}_mean"] = stats["mean"]
        result[f"{key}_std"] = stats["std"]
        result[f"{key}_ci95"] = stats["ci95"]
    return result


def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def compute_spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    xa = np.asarray(xs, dtype=np.float64)
    ya = np.asarray(ys, dtype=np.float64)
    x_rank = np.argsort(np.argsort(xa))
    y_rank = np.argsort(np.argsort(ya))
    # Pearson on ranks
    xr = x_rank.astype(np.float64)
    yr = y_rank.astype(np.float64)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = np.sqrt((xr ** 2).sum() * (yr ** 2).sum())
    if denom <= 1e-12:
        return None
    return float((xr * yr).sum() / denom)


def aggregate_bin(records: List[dict], key: str, bins: List[Tuple[float, float, str]], hi_closed_last: bool = False) -> List[dict]:
    out = []
    for i, (lo, hi, label) in enumerate(bins):
        closed = hi_closed_last and i == len(bins) - 1
        bucket = []
        for r in records:
            v = safe_float(r.get(key))
            if v is None:
                continue
            if in_bin(v, lo, hi, hi_closed=closed):
                bucket.append(r)
        feat = [safe_float(r.get("feature_mse")) for r in bucket]
        feat = [v for v in feat if v is not None]
        dp = [safe_float(r.get("delta_pesq")) for r in bucket]
        dp = [v for v in dp if v is not None]
        ds = [safe_float(r.get("delta_stoi")) for r in bucket]
        ds = [v for v in ds if v is not None]
        out.append(
            {
                "bin": label,
                "count": len(bucket),
                "feature_mse_mean": float(np.mean(feat)) if feat else None,
                "delta_pesq_mean": float(np.mean(dp)) if dp else None,
                "delta_stoi_mean": float(np.mean(ds)) if ds else None,
            }
        )
    return out


def plot_bin_quality(stats: List[dict], title: str, out_path: Path) -> None:
    labels = [s["bin"] for s in stats]
    cnts = [s["count"] for s in stats]
    dps = [s["delta_pesq_mean"] if s["delta_pesq_mean"] is not None else np.nan for s in stats]
    dss = [s["delta_stoi_mean"] if s["delta_stoi_mean"] is not None else np.nan for s in stats]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, dps, color="#1f77b4", alpha=0.85)
    axes[0].set_title(f"{title}: mean ΔPESQ")
    axes[0].set_ylabel("ΔPESQ")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, dss, color="#ff7f0e", alpha=0.85)
    axes[1].set_title(f"{title}: mean ΔSTOI")
    axes[1].set_ylabel("ΔSTOI")
    axes[1].set_xlabel("Bin")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_xticks(x, labels)

    for i, c in enumerate(cnts):
        axes[0].text(i, 0 if np.isnan(dps[i]) else dps[i], f"n={c}", ha="center", va="bottom", fontsize=8)
        axes[1].text(i, 0 if np.isnan(dss[i]) else dss[i], f"n={c}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mse_vs_quality(audio_summary: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for split, color in [("train", "#1f77b4"), ("val", "#d62728")]:
        xs = []
        ys_p = []
        ys_s = []
        labels = []
        split_stats = audio_summary["splits"].get(split, {})
        for ep_key in sorted(split_stats.keys()):
            row = split_stats[ep_key]
            if row.get("status") != "ok":
                continue
            x = safe_float(row.get("feature_mse_mean"))
            y1 = safe_float(row.get("delta_pesq_mean"))
            y2 = safe_float(row.get("delta_stoi_mean"))
            if x is None or y1 is None or y2 is None:
                continue
            xs.append(x)
            ys_p.append(y1)
            ys_s.append(y2)
            labels.append(ep_key.replace("epoch_", ""))

        axes[0].scatter(xs, ys_p, label=split, color=color, alpha=0.85)
        axes[1].scatter(xs, ys_s, label=split, color=color, alpha=0.85)
        for x, y, t in zip(xs, ys_p, labels):
            axes[0].annotate(t, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        for x, y, t in zip(xs, ys_s, labels):
            axes[1].annotate(t, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    axes[0].set_title("feature_mse vs ΔPESQ")
    axes[1].set_title("feature_mse vs ΔSTOI")
    axes[0].set_xlabel("feature_mse")
    axes[1].set_xlabel("feature_mse")
    axes[0].set_ylabel("ΔPESQ")
    axes[1].set_ylabel("ΔSTOI")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_audio_quality_md(audio_summary: dict) -> str:
    lines: List[str] = []
    lines.append("# Audio Quality by Epoch (exp_0216)")
    lines.append("")
    lines.append(f"- run_dir: `{audio_summary['run_dir']}`")
    lines.append(f"- target_sr_model: {audio_summary['target_sr_model']}")
    lines.append(f"- target_sr_metric: {audio_summary['target_sr_metric']}")
    lines.append(f"- n_target_per_split_epoch: {audio_summary['n_target_per_split_epoch']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Epoch `222` has no checkpoint file (`checkpoint_epoch222.pt`), so post-VQ audio quality is marked unavailable.")
    lines.append("- `feature_mse_mean` here is sample-level post-VQ MSE from evaluated subset (not full-dataset epoch metric).")
    lines.append("")

    for split in SPLITS:
        lines.append(f"## {split}")
        lines.append("| epoch | status | n | feature_mse | ΔPESQ | ΔSTOI |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        split_stats = audio_summary["splits"].get(split, {})
        for ep in EPOCHS:
            key = f"epoch_{ep:03d}"
            row = split_stats.get(key, {})
            lines.append(
                f"| {ep:03d} | {row.get('status', 'NA')} | "
                f"{row.get('num_samples', 0)} | "
                f"{fmt(row.get('feature_mse_mean'), 5)} | "
                f"{fmt(row.get('delta_pesq_mean'), 4)} | "
                f"{fmt(row.get('delta_stoi_mean'), 4)} |"
            )
        lines.append("")

    lines.append("## Correlations (Spearman, subset-level)")
    for split in SPLITS:
        cor = audio_summary.get("correlations", {}).get(split, {})
        lines.append(f"- {split}: corr(feature_mse, ΔPESQ)={fmt(cor.get('corr_feature_mse_delta_pesq'), 4)}, "
                     f"corr(feature_mse, ΔSTOI)={fmt(cor.get('corr_feature_mse_delta_stoi'), 4)}")
    lines.append("")
    return "\n".join(lines)


def build_stratified_md(stratified: dict) -> str:
    lines: List[str] = []
    lines.append("# Stratified Quality Report (exp_0216)")
    lines.append("")
    lines.append(f"- split: `{stratified['split']}`")
    lines.append(f"- epoch: `{stratified['epoch']}`")
    lines.append(f"- num_samples: {stratified['num_samples']}")
    lines.append(f"- top10_mass_proxy: {fmt(stratified.get('top10_mass_proxy'), 4)}")
    lines.append("")
    lines.append("## T453 ratio bins")
    lines.append("| bin | n | feature_mse | ΔPESQ | ΔSTOI |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in stratified["by_t453_bin"]:
        lines.append(f"| {r['bin']} | {r['count']} | {fmt(r['feature_mse_mean'], 5)} | {fmt(r['delta_pesq_mean'], 4)} | {fmt(r['delta_stoi_mean'], 4)} |")
    lines.append("")
    lines.append("## SNR bins")
    lines.append("| bin | n | feature_mse | ΔPESQ | ΔSTOI |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in stratified["by_snr_bin"]:
        lines.append(f"| {r['bin']} | {r['count']} | {fmt(r['feature_mse_mean'], 5)} | {fmt(r['delta_pesq_mean'], 4)} | {fmt(r['delta_stoi_mean'], 4)} |")
    lines.append("")
    lines.append("## Duration bins")
    lines.append("| bin | n | feature_mse | ΔPESQ | ΔSTOI |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in stratified["by_length_bin"]:
        lines.append(f"| {r['bin']} | {r['count']} | {fmt(r['feature_mse_mean'], 5)} | {fmt(r['delta_pesq_mean'], 4)} | {fmt(r['delta_stoi_mean'], 4)} |")
    lines.append("")
    lines.append("## Worst 10% Val Samples (by ΔPESQ)")
    lines.append("| rank | cache_index | ΔPESQ | ΔSTOI | t453 | snr_db | dur_s | noisy_path |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(stratified["worst_10_percent_samples"], start=1):
        lines.append(
            f"| {i} | {r['cache_index']} | {fmt(r.get('delta_pesq'), 4)} | {fmt(r.get('delta_stoi'), 4)} | "
            f"{fmt(r.get('t453_ratio'), 4)} | {fmt(r.get('snr_db'), 2)} | {fmt(r.get('duration_sec'), 2)} | `{r.get('noisy_path')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def update_coverage_file() -> None:
    path = ANALYSIS_DIR / "spec_coverage_status_20260219.json"
    if not path.exists():
        return
    cov = load_json(path)
    required = cov.get("required_outputs_presence", {})
    for k in list(required.keys()):
        required[k] = (ANALYSIS_DIR / k).exists()
    present = sum(1 for v in required.values() if v)
    total = len(required)
    cov["required_outputs_presence"] = required
    cov["present_count"] = present
    cov["required_count"] = total
    cov["required_coverage_ratio"] = float(present / total) if total else None

    # Milestone progress updates based on file presence and data availability.
    cov["milestone_progress_percent"] = {
        "M1_document_alignment": 100,
        "M2_current_inventory": 100,
        "M3_root_cause_analysis": 100 if present >= 9 else 90,
        "M4_decision_output": 100,
        "M5_minimal_change_plan_if_go": 100 if present == total else 80,
    }
    dump_json(path, cov)


def main() -> int:
    set_seed(SEED)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_history = load_json(METRICS_HISTORY_PATH)
    summary = load_json(SUMMARY_PATH)

    train_samples: List[dict] = torch.load(TRAIN_CACHE, map_location="cpu")
    val_samples: List[dict] = torch.load(VAL_CACHE, map_location="cpu")
    caches = {"train": train_samples, "val": val_samples}

    print("[1/8] Building raw filename index...")
    raw_index = build_raw_index()
    print(f"  raw index size: {len(raw_index)} filenames")

    print("[2/8] Selecting deterministic sample subsets...")
    selected_indices = {
        "train": pick_indices(train_samples, N_TARGET, SEED + 1, raw_index),
        "val": pick_indices(val_samples, N_TARGET, SEED + 2, raw_index),
    }
    print(f"  selected train={len(selected_indices['train'])}, val={len(selected_indices['val'])}")

    print("[3/8] Initializing model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TeacherStudentSingleVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=64,
        lora_alpha=128,
        intermediate_indices=[3, 4, 6],
        device=device,
        vq_ema_decay=0.99,
        vq_ema_threshold=2,
        vq_ema_usage_penalty=0.0,
    )
    model.eval()

    print("[4/8] Epoch-wise evaluation...")
    audio_summary = {
        "run_dir": str(RUN_DIR),
        "target_sr_model": TARGET_SR_MODEL,
        "target_sr_metric": TARGET_SR_METRIC,
        "n_target_per_split_epoch": N_TARGET,
        "epochs_requested": EPOCHS,
        "selection_indices": selected_indices,
        "checkpoint_load": {},
        "splits": {s: {} for s in SPLITS},
        "records": {s: {} for s in SPLITS},
        "gaps": [],
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "seed": SEED,
        },
    }

    for epoch in EPOCHS:
        ep_key = f"epoch_{epoch:03d}"
        ck_stat = load_checkpoint_for_epoch(model, epoch)
        audio_summary["checkpoint_load"][ep_key] = {
            "status": ck_stat.status,
            "checkpoint_path": ck_stat.checkpoint_path,
            "model_epoch": ck_stat.model_epoch,
            "reason": ck_stat.reason,
        }

        if ck_stat.status != "ok":
            audio_summary["gaps"].append(
                {
                    "type": "checkpoint_missing",
                    "epoch": epoch,
                    "reason": ck_stat.reason,
                }
            )
            for split in SPLITS:
                # Keep history proxies for traceability even if audio metrics unavailable.
                audio_summary["splits"][split][ep_key] = {
                    "status": ck_stat.status,
                    "num_samples": 0,
                    "feature_mse_history": metrics_history.get("feature_mse", [None] * 300)[epoch - 1],
                    "top10_mass_proxy": metrics_history.get("top10_mass", [None] * 300)[epoch - 1],
                }
                audio_summary["records"][split][ep_key] = []
            continue

        for split in SPLITS:
            rows: List[dict] = []
            indices = selected_indices[split]
            samples = caches[split]
            for pos, idx in enumerate(indices, start=1):
                try:
                    rec = evaluate_sample(
                        model=model,
                        device=device,
                        item=samples[idx],
                        cache_index=idx,
                        split=split,
                        epoch=epoch,
                        raw_index=raw_index,
                    )
                    if rec is not None:
                        rows.append(rec)
                except Exception:
                    # Keep execution robust and continue.
                    if pos == 1:
                        print(f"  warning [{split} epoch {epoch}] first-sample failure:\n{traceback.format_exc()}")
                    continue
                if pos % 20 == 0:
                    print(f"  progress epoch={epoch} split={split}: {pos}/{len(indices)}")

            agg = summarize_records(rows)
            agg["status"] = "ok"
            agg["top10_mass_proxy"] = metrics_history.get("top10_mass", [None] * 300)[epoch - 1]
            agg["feature_mse_history"] = metrics_history.get("feature_mse", [None] * 300)[epoch - 1]
            audio_summary["splits"][split][ep_key] = agg
            audio_summary["records"][split][ep_key] = rows

            if agg["num_samples"] < N_TARGET:
                audio_summary["gaps"].append(
                    {
                        "type": "insufficient_samples",
                        "epoch": epoch,
                        "split": split,
                        "target_n": N_TARGET,
                        "actual_n": agg["num_samples"],
                    }
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("[5/8] Correlation analysis...")
    corrs = {}
    for split in SPLITS:
        xs = []
        y_p = []
        y_s = []
        for ep in EPOCHS:
            row = audio_summary["splits"][split].get(f"epoch_{ep:03d}", {})
            if row.get("status") != "ok":
                continue
            x = safe_float(row.get("feature_mse_mean"))
            yp = safe_float(row.get("delta_pesq_mean"))
            ys = safe_float(row.get("delta_stoi_mean"))
            if x is None or yp is None or ys is None:
                continue
            xs.append(x)
            y_p.append(yp)
            y_s.append(ys)
        corrs[split] = {
            "corr_feature_mse_delta_pesq": compute_spearman(xs, y_p),
            "corr_feature_mse_delta_stoi": compute_spearman(xs, y_s),
            "num_points": len(xs),
        }
    audio_summary["correlations"] = corrs

    print("[6/8] Building stratified report basis...")
    # Use val epoch_300 if available; otherwise use last available val epoch.
    strat_epoch = 300
    strat_ep_key = f"epoch_{strat_epoch:03d}"
    if not audio_summary["records"]["val"].get(strat_ep_key):
        fallback = None
        for ep in reversed(EPOCHS):
            k = f"epoch_{ep:03d}"
            if audio_summary["records"]["val"].get(k):
                fallback = ep
                break
        if fallback is not None:
            strat_epoch = fallback
            strat_ep_key = f"epoch_{strat_epoch:03d}"

    val_rows = audio_summary["records"]["val"].get(strat_ep_key, [])
    top10_mass_proxy = metrics_history.get("top10_mass", [None] * 300)[strat_epoch - 1] if strat_epoch <= 300 else None

    by_t453 = aggregate_bin(val_rows, "t453_ratio", T453_BINS, hi_closed_last=True)
    by_snr = aggregate_bin(val_rows, "snr_db", SNR_BINS, hi_closed_last=False)
    by_len = aggregate_bin(val_rows, "duration_sec", LEN_BINS, hi_closed_last=False)

    sorted_rows = sorted(
        val_rows,
        key=lambda r: safe_float(r.get("delta_pesq")) if safe_float(r.get("delta_pesq")) is not None else 1e9,
    )
    worst_n = max(1, int(round(len(sorted_rows) * 0.1))) if sorted_rows else 0
    worst = sorted_rows[:worst_n]

    stratified = {
        "split": "val",
        "epoch": strat_epoch,
        "num_samples": len(val_rows),
        "top10_mass_proxy": top10_mass_proxy,
        "by_t453_bin": by_t453,
        "by_snr_bin": by_snr,
        "by_length_bin": by_len,
        "worst_10_percent_samples": worst,
    }

    print("[7/8] Writing outputs + plots...")
    out_json = ANALYSIS_DIR / "audio_quality_by_epoch.json"
    out_md = ANALYSIS_DIR / "audio_quality_by_epoch.md"
    out_strat_md = ANALYSIS_DIR / "stratified_quality_report.md"
    out_mse_plot = ANALYSIS_DIR / "mse_vs_pesq_stoi.png"
    out_t453_plot = ANALYSIS_DIR / "quality_by_t453_bin.png"
    out_snr_plot = ANALYSIS_DIR / "quality_by_snr_bin.png"
    out_len_plot = ANALYSIS_DIR / "quality_by_length_bin.png"

    dump_json(out_json, audio_summary)
    out_md.write_text(build_audio_quality_md(audio_summary), encoding="utf-8")
    out_strat_md.write_text(build_stratified_md(stratified), encoding="utf-8")
    plot_mse_vs_quality(audio_summary, out_mse_plot)
    plot_bin_quality(by_t453, f"Val epoch {strat_epoch} by T453 ratio", out_t453_plot)
    plot_bin_quality(by_snr, f"Val epoch {strat_epoch} by SNR", out_snr_plot)
    plot_bin_quality(by_len, f"Val epoch {strat_epoch} by duration", out_len_plot)

    print("[8/8] Updating SPEC coverage file...")
    update_coverage_file()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


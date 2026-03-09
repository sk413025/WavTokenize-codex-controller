#!/usr/bin/env python3
"""
Evaluate supplemental replay checkpoint (epoch222_replay) on fixed N=100 indices.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/sbplab/ruizi/WavTokenizer-main")

from families.deps.wavtokenizer_core.config import WAVTOK_CKPT, WAVTOK_CONFIG  # noqa: E402
from families.deps.encoder_vq_core.models_single_vq import TeacherStudentSingleVQ  # noqa: E402


ANALYSIS_DIR = ROOT / "exp_0217" / "analysis_commit_5e859b0"
BASE_QUALITY_JSON = ANALYSIS_DIR / "audio_quality_by_epoch.json"
REPLAY_CKPT = ANALYSIS_DIR / "replay_from220_to222" / "checkpoints" / "checkpoint_epoch222_replay.pt"
OUT_JSON = ANALYSIS_DIR / "audio_quality_epoch222_replay.json"
OUT_MD = ANALYSIS_DIR / "audio_quality_epoch222_replay.md"

TRAIN_CACHE = ROOT / "data" / "train_cache_filtered.pt"
VAL_CACHE = ROOT / "data" / "val_cache_filtered.pt"

RAW_ROOT = Path("/home/sbplab/ruizi/WavTokenize/data/raw")
CLEAN_ROOT = Path("/home/sbplab/ruizi/WavTokenize/data/clean/box2")

TARGET_SR_MODEL = 24000
TARGET_SR_METRIC = 16000


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(path_or_name: str, raw_index: Dict[str, List[Path]]) -> Optional[Path]:
    p = Path(path_or_name)
    if p.is_absolute() and p.exists():
        return p
    name = p.name
    if "_clean_" in name:
        c = CLEAN_ROOT / name
        if c.exists():
            return c
    for mat in ("box", "papercup", "plastic"):
        c = RAW_ROOT / mat / name
        if c.exists():
            return c
    m = raw_index.get(name, [])
    return m[0] if m else None


def build_raw_index() -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in RAW_ROOT.rglob("*.wav"):
        idx.setdefault(p.name, []).append(p)
    return idx


def load_mono(path: Path) -> Tuple[np.ndarray, int]:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return a[:, 0], int(sr)


def rs(a: np.ndarray, src: int, tgt: int) -> np.ndarray:
    if src == tgt:
        return a.astype(np.float32, copy=False)
    return resample_poly(a, tgt, src).astype(np.float32)


def clip(a: np.ndarray) -> np.ndarray:
    return np.clip(a, -1.0, 1.0).astype(np.float32, copy=False)


def stats(values: List[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "ci95": None}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci}


def main():
    base = load_json(BASE_QUALITY_JSON)
    indices = base["selection_indices"]
    train_samples = torch.load(TRAIN_CACHE, map_location="cpu")
    val_samples = torch.load(VAL_CACHE, map_location="cpu")
    caches = {"train": train_samples, "val": val_samples}

    raw_index = build_raw_index()

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
    ck = torch.load(REPLAY_CKPT, map_location="cpu")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in ck["lora_state"]:
                p.copy_(ck["lora_state"][n].to(p.device, dtype=p.dtype))
    model.vq.load_state_dict(ck["vq_state_dict"])
    model.eval()

    out = {
        "checkpoint": str(REPLAY_CKPT),
        "note": "supplemental replay checkpoint from epoch220->222",
        "splits": {},
        "records": {},
    }

    for split in ("train", "val"):
        rows = []
        for idx in indices[split]:
            item = caches[split][idx]
            noisy_p = resolve_path(str(item.get("noisy_path", "")), raw_index)
            clean_p = resolve_path(str(item.get("clean_path", "")), raw_index)
            if noisy_p is None or clean_p is None:
                continue
            noisy_raw, sr_n = load_mono(noisy_p)
            clean_raw, sr_c = load_mono(clean_p)
            noisy24 = rs(noisy_raw, sr_n, TARGET_SR_MODEL)
            clean24 = rs(clean_raw, sr_c, TARGET_SR_MODEL)
            n = min(len(noisy24), len(clean24))
            if n <= TARGET_SR_MODEL // 10:
                continue
            noisy24 = clip(noisy24[:n])
            clean24 = clip(clean24[:n])

            with torch.no_grad():
                noisy_t = torch.from_numpy(noisy24).to(device).view(1, 1, -1)
                clean_t = torch.from_numpy(clean24).to(device).view(1, 1, -1)
                o = model(clean_t, noisy_t)
                rec_t = model.decode(o["student_quantized"])
                if rec_t.dim() == 3:
                    rec_t = rec_t.squeeze(1)
                rec24 = rec_t.squeeze(0).float().cpu().numpy()
                fmse = float(torch.mean((o["student_quantized"] - o["teacher_encoder_out"]) ** 2).item())

            clean16 = rs(clean24, TARGET_SR_MODEL, TARGET_SR_METRIC)
            noisy16 = rs(noisy24, TARGET_SR_MODEL, TARGET_SR_METRIC)
            rec16 = rs(rec24, TARGET_SR_MODEL, TARGET_SR_METRIC)
            m = min(len(clean16), len(noisy16), len(rec16))
            clean16, noisy16, rec16 = clip(clean16[:m]), clip(noisy16[:m]), clip(rec16[:m])
            if m <= TARGET_SR_METRIC // 10:
                continue

            try:
                p_noisy = float(pesq(TARGET_SR_METRIC, clean16, noisy16, "wb"))
                p_rec = float(pesq(TARGET_SR_METRIC, clean16, rec16, "wb"))
                s_noisy = float(stoi(clean16, noisy16, TARGET_SR_METRIC, extended=False))
                s_rec = float(stoi(clean16, rec16, TARGET_SR_METRIC, extended=False))
            except Exception:
                continue

            rows.append(
                {
                    "cache_index": int(idx),
                    "feature_mse": fmse,
                    "pesq_noisy": p_noisy,
                    "pesq_recon": p_rec,
                    "delta_pesq": p_rec - p_noisy,
                    "stoi_noisy": s_noisy,
                    "stoi_recon": s_rec,
                    "delta_stoi": s_rec - s_noisy,
                }
            )

        out["records"][split] = rows
        out["splits"][split] = {
            "num_samples": len(rows),
            "feature_mse": stats([r["feature_mse"] for r in rows]),
            "delta_pesq": stats([r["delta_pesq"] for r in rows]),
            "delta_stoi": stats([r["delta_stoi"] for r in rows]),
            "pesq_noisy": stats([r["pesq_noisy"] for r in rows]),
            "pesq_recon": stats([r["pesq_recon"] for r in rows]),
            "stoi_noisy": stats([r["stoi_noisy"] for r in rows]),
            "stoi_recon": stats([r["stoi_recon"] for r in rows]),
        }

    dump_json(OUT_JSON, out)

    lines = [
        "# Replay Epoch222 Audio Quality (Supplemental)",
        "",
        f"- checkpoint: `{REPLAY_CKPT}`",
        "- note: replay artifact from epoch220 checkpoint; not original run artifact",
        "",
        "| split | n | feature_mse | ΔPESQ | ΔSTOI |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        s = out["splits"][split]
        lines.append(
            f"| {split} | {s['num_samples']} | "
            f"{s['feature_mse']['mean']:.5f} | {s['delta_pesq']['mean']:.4f} | {s['delta_stoi']['mean']:.4f} |"
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()

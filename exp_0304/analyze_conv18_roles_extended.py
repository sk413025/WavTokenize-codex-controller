#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decoder.pretrained import WavTokenizer
from encoder.modules.conv import SConv1d
from encoder.modules.seanet import SEANetResnetBlock


DEFAULT_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
DEFAULT_CKPT = "/home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt"
DEFAULT_OUTDIR = "/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0304/wavtokenizer_featuremap_14wav_extended"


FILES = [
    # Utterance 001 (existing baseline 6 files)
    {
        "id": "boy4_clean_001",
        "speaker": "boy4",
        "utt": "001",
        "kind": "clean",
        "material": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy4_clean_001.wav",
    },
    {
        "id": "boy7_clean_001",
        "speaker": "boy7",
        "utt": "001",
        "kind": "clean",
        "material": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy7_clean_001.wav",
    },
    {
        "id": "girl9_clean_001",
        "speaker": "girl9",
        "utt": "001",
        "kind": "clean",
        "material": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean_mix_test/recon/0224a/snr+10dB/nor_girl9_clean_001.wav",
    },
    {
        "id": "boy4_box_001",
        "speaker": "boy4",
        "utt": "001",
        "kind": "ldv",
        "material": "box",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy4/box/nor_boy4_box_LDV_001.wav",
    },
    {
        "id": "boy7_box_001",
        "speaker": "boy7",
        "utt": "001",
        "kind": "ldv",
        "material": "box",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy7/box/nor_boy7_box_LDV_001.wav",
    },
    {
        "id": "girl9_box_001",
        "speaker": "girl9",
        "utt": "001",
        "kind": "ldv",
        "material": "box",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/recon/0224a/girl9/box/nor_girl9_box_LDV_001.wav",
    },
    # Utterance 002 (new files user provided)
    {
        "id": "boy4_clean_002",
        "speaker": "boy4",
        "utt": "002",
        "kind": "clean",
        "material": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy4_clean_002.wav",
    },
    {
        "id": "boy7_clean_002",
        "speaker": "boy7",
        "utt": "002",
        "kind": "clean",
        "material": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy7_clean_002.wav",
    },
    {
        "id": "girl9_clean_002",
        "speaker": "girl9",
        "utt": "002",
        "kind": "clean",
        "material": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_girl9_clean_002.wav",
    },
    {
        "id": "boy4_box_002",
        "speaker": "boy4",
        "utt": "002",
        "kind": "ldv",
        "material": "box",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy4/box/nor_boy4_box_LDV_002.wav",
    },
    {
        "id": "boy7_box_002",
        "speaker": "boy7",
        "utt": "002",
        "kind": "ldv",
        "material": "box",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy7/box/nor_boy7_box_LDV_002.wav",
    },
    {
        "id": "girl9_box_002",
        "speaker": "girl9",
        "utt": "002",
        "kind": "ldv",
        "material": "box",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/girl9/box/nor_girl9_box_LDV_002.wav",
    },
    # Additional materials
    {
        "id": "boy7_papercup_001",
        "speaker": "boy7",
        "utt": "001",
        "kind": "ldv",
        "material": "papercup",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy7/papercup/nor_boy7_papercup_LDV_001.wav",
    },
    {
        "id": "girl9_plastic_001",
        "speaker": "girl9",
        "utt": "001",
        "kind": "ldv",
        "material": "plastic",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/girl9/plastic/nor_girl9_plastic_LDV_001.wav",
    },
]


@dataclass
class ConvLayerSpec:
    idx: int
    module_path: str
    stage: str
    module: torch.nn.Module


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def read_audio_24k_mono(path: str) -> torch.Tensor:
    wav_np, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = torch.from_numpy(wav_np).transpose(0, 1)  # [C, T]
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def build_conv18_specs(encoder: torch.nn.Module) -> List[ConvLayerSpec]:
    specs: List[ConvLayerSpec] = []
    idx = 0
    for t_i, mod in enumerate(encoder.model):
        if isinstance(mod, SConv1d):
            stage = "stem" if t_i == 0 else ("downsample" if t_i in [3, 6, 9, 12] else "projection")
            specs.append(
                ConvLayerSpec(
                    idx=idx,
                    module_path=f"encoder.model[{t_i}]",
                    stage=stage,
                    module=mod,
                )
            )
            idx += 1
        elif isinstance(mod, SEANetResnetBlock):
            c1 = mod.block[1]
            c2 = mod.block[3]
            sc = mod.shortcut
            assert isinstance(c1, SConv1d)
            assert isinstance(c2, SConv1d)
            specs.append(ConvLayerSpec(idx=idx, module_path=f"encoder.model[{t_i}].block[1]", stage="residual", module=c1))
            idx += 1
            specs.append(ConvLayerSpec(idx=idx, module_path=f"encoder.model[{t_i}].block[3]", stage="residual", module=c2))
            idx += 1
            if isinstance(sc, SConv1d):
                specs.append(ConvLayerSpec(idx=idx, module_path=f"encoder.model[{t_i}].shortcut", stage="residual", module=sc))
                idx += 1
    if len(specs) != 18:
        raise RuntimeError(f"Expected 18 conv-like layers but got {len(specs)}.")
    return specs


def extract_layer_outputs(
    model: WavTokenizer, conv_specs: List[ConvLayerSpec], wav: torch.Tensor, device: torch.device
) -> Dict[int, torch.Tensor]:
    caches: Dict[int, torch.Tensor] = {}
    hooks = []
    for spec in conv_specs:
        def _hook_factory(li: int):
            def _hook(_m, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                caches[li] = out.detach().cpu().squeeze(0)  # [C, T]
            return _hook
        hooks.append(spec.module.register_forward_hook(_hook_factory(spec.idx)))
    wav = wav.to(device)
    bw = torch.tensor([0], device=device)
    with torch.no_grad():
        _features, _codes = model.encode_infer(wav, bandwidth_id=bw)
    for h in hooks:
        h.remove()
    return caches


def make_scalars(layer_out: torch.Tensor) -> Dict[str, float]:
    abs_mean = float(layer_out.abs().mean().item())
    rms = float(torch.sqrt((layer_out ** 2).mean()).item())
    temp_std = float(layer_out.std(dim=-1, unbiased=False).mean().item())
    if layer_out.shape[-1] > 1:
        delta = layer_out[:, 1:] - layer_out[:, :-1]
        delta_rms = float(torch.sqrt((delta ** 2).mean()).item())
    else:
        delta_rms = 0.0
    return {"abs_mean": abs_mean, "rms": rms, "temp_std": temp_std, "delta_rms": delta_rms}


def nearest_clean_ref(file_item: Dict, clean_items_by_speaker: Dict[str, List[Dict]]) -> Dict:
    spk = file_item["speaker"]
    utt = int(file_item["utt"])
    cands = clean_items_by_speaker[spk]
    best = min(cands, key=lambda x: abs(int(x["utt"]) - utt))
    return best


def build_pairs(files: List[Dict]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    # same speaker clean-vs-material pairs
    clean_items = [x for x in files if x["kind"] == "clean"]
    clean_by_spk: Dict[str, List[Dict]] = {}
    for c in clean_items:
        clean_by_spk.setdefault(c["speaker"], []).append(c)

    same_pairs = []
    for f in files:
        if f["kind"] == "clean":
            continue
        ref = nearest_clean_ref(f, clean_by_spk)
        same_pairs.append((ref["id"], f["id"]))

    # cross-speaker same material+utt pairs
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for f in files:
        key = (f["material"], f["utt"])
        groups.setdefault(key, []).append(f)

    cross_pairs = []
    for _key, items in groups.items():
        for a, b in itertools.combinations(items, 2):
            if a["speaker"] != b["speaker"]:
                cross_pairs.append((a["id"], b["id"]))
    return same_pairs, cross_pairs


def plot_metric(
    outpath: Path,
    conv_count: int,
    files: List[Dict],
    per_file_scalars: Dict[str, Dict[int, Dict[str, float]]],
    metric_key: str,
    ylabel: str,
):
    x = list(range(conv_count))
    labels = [f"L{i:02d}" for i in x]
    plt.figure(figsize=(16, 6))

    color_map = {"boy4": "#1f77b4", "boy7": "#2ca02c", "girl9": "#ff7f0e"}
    ls_map = {"clean": "-", "box": "--", "papercup": "-.", "plastic": ":"}
    marker_map = {"001": "o", "002": "s"}

    # fixed order for readability
    sorted_files = sorted(files, key=lambda z: (z["speaker"], z["utt"], z["material"]))
    for f in sorted_files:
        fid = f["id"]
        y = [per_file_scalars[fid][i][metric_key] for i in x]
        plt.plot(
            x, y,
            color=color_map.get(f["speaker"], "#333333"),
            linestyle=ls_map.get(f["material"], "-"),
            marker=marker_map.get(f["utt"], "o"),
            linewidth=1.7,
            markersize=3,
            label=f"{fid}",
        )

    plt.xticks(x, labels, rotation=45, ha="right", fontsize=9)
    plt.xlabel("18 conv-like layers")
    plt.ylabel(ylabel)
    plt.title(f"{metric_key} across 14 files (speaker=color, material=linestyle, utt=marker)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = WavTokenizer.from_pretrained0802(args.config, args.ckpt).to(device).eval()
    encoder = model.feature_extractor.encodec.encoder
    conv_specs = build_conv18_specs(encoder)

    per_file_vecs: Dict[str, Dict[int, torch.Tensor]] = {}
    per_file_scalars: Dict[str, Dict[int, Dict[str, float]]] = {}
    file_meta = {x["id"]: x for x in FILES}

    for item in FILES:
        fid = item["id"]
        wav = read_audio_24k_mono(item["path"])
        raw = extract_layer_outputs(model, conv_specs, wav, device)
        per_file_vecs[fid] = {}
        per_file_scalars[fid] = {}
        for li, out in raw.items():
            per_file_vecs[fid][li] = out.mean(dim=-1).float()  # [C]
            per_file_scalars[fid][li] = make_scalars(out)

    same_pairs, cross_pairs = build_pairs(FILES)

    # Role metrics
    temp_all = [per_file_scalars[fid][li]["temp_std"] for fid in per_file_scalars for li in per_file_scalars[fid]]
    tmin, tmax = float(min(temp_all)), float(max(temp_all))
    tspan = max(tmax - tmin, 1e-8)

    role_rows = []
    for li in range(18):
        same_cos_vals = [cos(per_file_vecs[a][li], per_file_vecs[b][li]) for a, b in same_pairs]
        cross_cos_vals = [cos(per_file_vecs[a][li], per_file_vecs[b][li]) for a, b in cross_pairs]
        same_cos = float(np.mean(same_cos_vals)) if same_cos_vals else 0.0
        cross_cos = float(np.mean(cross_cos_vals)) if cross_cos_vals else 0.0

        noise_sens = float(1.0 - same_cos)
        speaker_identity = float(max(0.0, same_cos - cross_cos))
        content_shared = float(max(0.0, cross_cos))
        temp_mean = float(np.mean([per_file_scalars[fid][li]["temp_std"] for fid in per_file_scalars]))
        temp_norm = float((temp_mean - tmin) / tspan)

        role_rows.append(
            {
                "layer_idx": li,
                "module_path": conv_specs[li].module_path,
                "stage": conv_specs[li].stage,
                "same_speaker_clean_material_cos": same_cos,
                "cross_speaker_same_material_utt_cos": cross_cos,
                "noise_sensitivity": noise_sens,
                "speaker_identity_score": speaker_identity,
                "content_shared_score": content_shared,
                "temporal_detail_score_norm": temp_norm,
            }
        )

    # LoRA recommendations
    adapt_rank = sorted(
        role_rows,
        key=lambda r: (0.70 * r["noise_sensitivity"] + 0.20 * r["temporal_detail_score_norm"] + 0.10 * (1.0 - r["content_shared_score"])),
        reverse=True,
    )
    freeze_rank = sorted(
        role_rows,
        key=lambda r: (0.60 * r["content_shared_score"] + 0.30 * r["speaker_identity_score"] + 0.10 * (1.0 - r["noise_sensitivity"])),
        reverse=True,
    )
    adapt_top6 = adapt_rank[:6]
    freeze_top6 = freeze_rank[:6]

    # Save per-file metrics
    metric_rows = []
    for f in FILES:
        fid = f["id"]
        for li in range(18):
            row = {
                "file_id": fid,
                "speaker": f["speaker"],
                "utt": f["utt"],
                "kind": f["kind"],
                "material": f["material"],
                "layer_idx": li,
                "module_path": conv_specs[li].module_path,
                "stage": conv_specs[li].stage,
            }
            row.update(per_file_scalars[fid][li])
            metric_rows.append(row)

    write_csv(
        outdir / "conv18_14wav_metrics.csv",
        metric_rows,
        [
            "file_id",
            "speaker",
            "utt",
            "kind",
            "material",
            "layer_idx",
            "module_path",
            "stage",
            "abs_mean",
            "rms",
            "temp_std",
            "delta_rms",
        ],
    )
    write_csv(
        outdir / "conv18_14wav_role_metrics.csv",
        role_rows,
        [
            "layer_idx",
            "module_path",
            "stage",
            "same_speaker_clean_material_cos",
            "cross_speaker_same_material_utt_cos",
            "noise_sensitivity",
            "speaker_identity_score",
            "content_shared_score",
            "temporal_detail_score_norm",
        ],
    )

    # save pair diagnostics
    pair_rows = []
    for a, b in same_pairs:
        pair_rows.append({"pair_type": "same_speaker_clean_material", "a": a, "b": b, "a_path": file_meta[a]["path"], "b_path": file_meta[b]["path"]})
    for a, b in cross_pairs:
        pair_rows.append({"pair_type": "cross_speaker_same_material_utt", "a": a, "b": b, "a_path": file_meta[a]["path"], "b_path": file_meta[b]["path"]})
    write_csv(outdir / "pair_diagnostics.csv", pair_rows, ["pair_type", "a", "b", "a_path", "b_path"])

    # plots
    plot_metric(
        outpath=outdir / "conv18_14wav_temp_std.png",
        conv_count=18,
        files=FILES,
        per_file_scalars=per_file_scalars,
        metric_key="temp_std",
        ylabel="Temporal Std (proxy)",
    )
    plot_metric(
        outpath=outdir / "conv18_14wav_delta_rms.png",
        conv_count=18,
        files=FILES,
        per_file_scalars=per_file_scalars,
        metric_key="delta_rms",
        ylabel="Frame-delta RMS (proxy)",
    )

    summary = {
        "num_files": len(FILES),
        "num_same_pairs": len(same_pairs),
        "num_cross_pairs": len(cross_pairs),
        "adapt_top6": [{"layer_idx": x["layer_idx"], "module_path": x["module_path"]} for x in adapt_top6],
        "freeze_top6": [{"layer_idx": x["layer_idx"], "module_path": x["module_path"]} for x in freeze_top6],
        "notes": {
            "same_pair_rule": "each non-clean file matched to nearest clean utterance of same speaker (prefer same utt)",
            "cross_pair_rule": "different speakers, same material and same utterance id",
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # human-readable report
    lines = []
    lines.append("Conv18 Extended Analysis (14 files)")
    lines.append(f"same_speaker_clean_material pairs: {len(same_pairs)}")
    lines.append(f"cross_speaker_same_material_utt pairs: {len(cross_pairs)}")
    lines.append("")
    lines.append("LoRA adapt_top6:")
    for x in adapt_top6:
        lines.append(
            f"- L{x['layer_idx']:02d} {x['module_path']}: noise={x['noise_sensitivity']:.3f}, "
            f"content={x['content_shared_score']:.3f}, temp={x['temporal_detail_score_norm']:.3f}"
        )
    lines.append("")
    lines.append("Freeze_top6:")
    for x in freeze_top6:
        lines.append(
            f"- L{x['layer_idx']:02d} {x['module_path']}: content={x['content_shared_score']:.3f}, "
            f"speaker={x['speaker_identity_score']:.3f}, noise={x['noise_sensitivity']:.3f}"
        )
    (outdir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] {outdir / 'conv18_14wav_temp_std.png'}")
    print(f"[OK] {outdir / 'conv18_14wav_delta_rms.png'}")
    print(f"[OK] {outdir / 'conv18_14wav_metrics.csv'}")
    print(f"[OK] {outdir / 'conv18_14wav_role_metrics.csv'}")
    print(f"[OK] {outdir / 'pair_diagnostics.csv'}")
    print(f"[OK] {outdir / 'summary.json'}")
    print(f"[OK] {outdir / 'report.txt'}")


if __name__ == "__main__":
    main()

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
from families.deps.no_vq_core.models_no_vq import TeacherStudentNoVQ


DEFAULT_BASE_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
DEFAULT_BASE_CKPT = "/home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt"
DEFAULT_0224A_CKPT = "/home/sbplab/ruizi/WavTokenize-feature-analysis/families/deps/no_vq_core/runs/no_vq_epoch_20260223_055458/best_model.pt"
DEFAULT_OUTDIR = "/home/sbplab/ruizi/WavTokenize-feature-analysis/families/official/material_generalization/wavtokenizer_featuremap_14wav_extended"


FILES = [
    {"id": "boy4_clean_001", "speaker": "boy4", "utt": "001", "kind": "clean", "material": "clean",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy4_clean_001.wav"},
    {"id": "boy7_clean_001", "speaker": "boy7", "utt": "001", "kind": "clean", "material": "clean",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy7_clean_001.wav"},
    {"id": "girl9_clean_001", "speaker": "girl9", "utt": "001", "kind": "clean", "material": "clean",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean_mix_test/recon/0224a/snr+10dB/nor_girl9_clean_001.wav"},
    {"id": "boy4_box_001", "speaker": "boy4", "utt": "001", "kind": "ldv", "material": "box",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy4/box/nor_boy4_box_LDV_001.wav"},
    {"id": "boy7_box_001", "speaker": "boy7", "utt": "001", "kind": "ldv", "material": "box",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy7/box/nor_boy7_box_LDV_001.wav"},
    {"id": "girl9_box_001", "speaker": "girl9", "utt": "001", "kind": "ldv", "material": "box",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/recon/0224a/girl9/box/nor_girl9_box_LDV_001.wav"},
    {"id": "boy4_clean_002", "speaker": "boy4", "utt": "002", "kind": "clean", "material": "clean",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy4_clean_002.wav"},
    {"id": "boy7_clean_002", "speaker": "boy7", "utt": "002", "kind": "clean", "material": "clean",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy7_clean_002.wav"},
    {"id": "girl9_clean_002", "speaker": "girl9", "utt": "002", "kind": "clean", "material": "clean",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_girl9_clean_002.wav"},
    {"id": "boy4_box_002", "speaker": "boy4", "utt": "002", "kind": "ldv", "material": "box",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy4/box/nor_boy4_box_LDV_002.wav"},
    {"id": "boy7_box_002", "speaker": "boy7", "utt": "002", "kind": "ldv", "material": "box",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy7/box/nor_boy7_box_LDV_002.wav"},
    {"id": "girl9_box_002", "speaker": "girl9", "utt": "002", "kind": "ldv", "material": "box",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/girl9/box/nor_girl9_box_LDV_002.wav"},
    {"id": "boy7_papercup_001", "speaker": "boy7", "utt": "001", "kind": "ldv", "material": "papercup",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy7/papercup/nor_boy7_papercup_LDV_001.wav"},
    {"id": "girl9_plastic_001", "speaker": "girl9", "utt": "001", "kind": "ldv", "material": "plastic",
     "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/girl9/plastic/nor_girl9_plastic_LDV_001.wav"},
]


@dataclass
class ConvLayerSpec:
    idx: int
    module_path: str
    module: torch.nn.Module


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
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
    specs = []
    idx = 0
    for t_i, mod in enumerate(encoder.model):
        if isinstance(mod, SConv1d):
            specs.append(ConvLayerSpec(idx=idx, module_path=f"encoder.model[{t_i}]", module=mod))
            idx += 1
        elif isinstance(mod, SEANetResnetBlock):
            c1 = mod.block[1]
            c2 = mod.block[3]
            sc = mod.shortcut
            assert isinstance(c1, SConv1d)
            assert isinstance(c2, SConv1d)
            specs.append(ConvLayerSpec(idx=idx, module_path=f"encoder.model[{t_i}].block[1]", module=c1))
            idx += 1
            specs.append(ConvLayerSpec(idx=idx, module_path=f"encoder.model[{t_i}].block[3]", module=c2))
            idx += 1
            if isinstance(sc, SConv1d):
                specs.append(ConvLayerSpec(idx=idx, module_path=f"encoder.model[{t_i}].shortcut", module=sc))
                idx += 1
    if len(specs) != 18:
        raise RuntimeError(f"Expected 18 conv-like layers, got {len(specs)}")
    return specs


def build_pairs(files: List[Dict]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    clean_items = [x for x in files if x["kind"] == "clean"]
    clean_by_spk: Dict[str, List[Dict]] = {}
    for c in clean_items:
        clean_by_spk.setdefault(c["speaker"], []).append(c)

    def nearest_clean_ref(f):
        spk = f["speaker"]
        utt = int(f["utt"])
        cands = clean_by_spk[spk]
        return min(cands, key=lambda x: abs(int(x["utt"]) - utt))

    same_pairs = []
    for f in files:
        if f["kind"] == "clean":
            continue
        ref = nearest_clean_ref(f)
        same_pairs.append((ref["id"], f["id"]))

    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for f in files:
        groups.setdefault((f["material"], f["utt"]), []).append(f)

    cross_pairs = []
    for _k, items in groups.items():
        for a, b in itertools.combinations(items, 2):
            if a["speaker"] != b["speaker"]:
                cross_pairs.append((a["id"], b["id"]))
    return same_pairs, cross_pairs


def extract_with_hooks(run_forward, conv_specs: List[ConvLayerSpec]) -> Dict[int, torch.Tensor]:
    caches: Dict[int, torch.Tensor] = {}
    hooks = []
    for spec in conv_specs:
        def _hook_factory(li):
            def _hook(_m, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                caches[li] = out.detach().cpu().squeeze(0)  # [C, T]
            return _hook
        hooks.append(spec.module.register_forward_hook(_hook_factory(spec.idx)))
    run_forward()
    for h in hooks:
        h.remove()
    return caches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base_ckpt", default=DEFAULT_BASE_CKPT)
    parser.add_argument("--lora_ckpt", default=DEFAULT_0224A_CKPT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    # Before (official)
    before_model = WavTokenizer.from_pretrained0802(args.base_config, args.base_ckpt).to(device).eval()
    before_encoder = before_model.feature_extractor.encodec.encoder
    before_specs = build_conv18_specs(before_encoder)

    # After (exp_0224a LoRA)
    lora_ckpt = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    lora_rank = int(lora_ckpt.get("config", {}).get("lora_rank", 64))
    lora_alpha = int(lora_ckpt.get("config", {}).get("lora_alpha", 128))
    after_model = TeacherStudentNoVQ(
        wavtok_config=args.base_config,
        wavtok_ckpt=args.base_ckpt,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        device=str(device),
    ).to(device).eval()
    missing, unexpected = after_model.load_state_dict(lora_ckpt["model_state_dict"], strict=False)
    print(f"[INFO] loaded 0224a ckpt, missing={len(missing)}, unexpected={len(unexpected)}")
    after_encoder = after_model.student.feature_extractor.encodec.encoder
    after_specs = build_conv18_specs(after_encoder)

    file_vec_before: Dict[str, Dict[int, torch.Tensor]] = {}
    file_vec_after: Dict[str, Dict[int, torch.Tensor]] = {}

    for f in FILES:
        wav = read_audio_24k_mono(f["path"])

        # before
        def run_before():
            with torch.no_grad():
                x = wav.to(device)
                bw = torch.tensor([0], device=device)
                before_model.encode_infer(x, bandwidth_id=bw)
        out_b = extract_with_hooks(run_before, before_specs)
        file_vec_before[f["id"]] = {li: out_b[li].mean(dim=-1).float() for li in out_b}

        # after (0224a student encoder)
        def run_after():
            with torch.no_grad():
                x = wav.to(device).unsqueeze(0)  # [B=1, C=1, T]
                after_model.student_extractor(x)
        out_a = extract_with_hooks(run_after, after_specs)
        file_vec_after[f["id"]] = {li: out_a[li].mean(dim=-1).float() for li in out_a}

    same_pairs, cross_pairs = build_pairs(FILES)

    rows = []
    for li in range(18):
        same_before = float(np.mean([cos(file_vec_before[a][li], file_vec_before[b][li]) for a, b in same_pairs]))
        cross_before = float(np.mean([cos(file_vec_before[a][li], file_vec_before[b][li]) for a, b in cross_pairs]))
        same_after = float(np.mean([cos(file_vec_after[a][li], file_vec_after[b][li]) for a, b in same_pairs]))
        cross_after = float(np.mean([cos(file_vec_after[a][li], file_vec_after[b][li]) for a, b in cross_pairs]))
        rows.append({
            "layer_idx": li,
            "same_before": same_before,
            "cross_before": cross_before,
            "same_after_0224a": same_after,
            "cross_after_0224a": cross_after,
            "delta_same": same_after - same_before,
            "delta_cross": cross_after - cross_before,
            "noise_before": 1.0 - same_before,
            "noise_after": 1.0 - same_after,
        })

    # save csv
    csv_path = outdir / "speaker_invariance_before_after_0224a.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    x = np.arange(18)
    same_b = np.array([r["same_before"] for r in rows])
    same_a = np.array([r["same_after_0224a"] for r in rows])
    cross_b = np.array([r["cross_before"] for r in rows])
    cross_a = np.array([r["cross_after_0224a"] for r in rows])
    delta_same = same_a - same_b

    # After-only figure (same style as original speaker_invariance_evidence)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x, same_a, "o-", label="same-speaker cosine (clean vs material) [0224a]", color="#d62728")
    axes[0].plot(x, cross_a, "s-", label="cross-speaker cosine (same material+utt) [0224a]", color="#1f77b4")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_title("Speaker Invariance Evidence (After LoRA: exp_0224a)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    sep_after = np.maximum(0.0, same_a - cross_a)
    axes[1].bar(x, sep_after, color=["#2ca02c" if v >= 0.05 else "#7f7f7f" for v in sep_after])
    axes[1].axhline(0.05, color="#2ca02c", linestyle="--", linewidth=1, label="separability ref=0.05")
    axes[1].set_ylabel("Speaker separability\nmax(0, same-cross)")
    axes[1].set_xlabel("Layer index (L00~L17)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"L{i:02d}" for i in x], rotation=45, ha="right")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    after_png = outdir / "speaker_invariance_evidence_0224a.png"
    plt.savefig(after_png, dpi=240)
    plt.close(fig)

    # Before/After comparison figure
    fig2, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax[0].plot(x, same_b, "o--", color="#ff9896", label="same before")
    ax[0].plot(x, same_a, "o-", color="#d62728", label="same after 0224a")
    ax[0].plot(x, cross_b, "s--", color="#9ecae1", label="cross before")
    ax[0].plot(x, cross_a, "s-", color="#1f77b4", label="cross after 0224a")
    ax[0].set_ylabel("Cosine")
    ax[0].set_title("Speaker Invariance: Before vs After (exp_0224a)")
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(ncol=2)
    ax[1].bar(x, delta_same, color=["#2ca02c" if v >= 0 else "#d62728" for v in delta_same], alpha=0.85)
    ax[1].axhline(0.0, color="black", linewidth=1)
    ax[1].set_ylabel("Delta same-speaker cosine\n(after - before)")
    ax[1].set_xlabel("Layer index (L00~L17)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"L{i:02d}" for i in x], rotation=45, ha="right")
    ax[1].grid(True, alpha=0.25)
    plt.tight_layout()
    compare_png = outdir / "speaker_invariance_before_after_0224a.png"
    plt.savefig(compare_png, dpi=240)
    plt.close(fig2)

    summary = {
        "same_pairs": len(same_pairs),
        "cross_pairs": len(cross_pairs),
        "lora_ckpt": args.lora_ckpt,
        "top_improved_same_cos": sorted(
            [{"layer_idx": r["layer_idx"], "delta_same": r["delta_same"]} for r in rows],
            key=lambda z: z["delta_same"],
            reverse=True,
        )[:6],
        "top_degraded_same_cos": sorted(
            [{"layer_idx": r["layer_idx"], "delta_same": r["delta_same"]} for r in rows],
            key=lambda z: z["delta_same"],
        )[:6],
    }
    summary_path = outdir / "speaker_invariance_before_after_0224a_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {after_png}")
    print(f"[OK] {compare_png}")
    print(f"[OK] {csv_path}")
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()

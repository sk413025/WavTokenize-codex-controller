#!/usr/bin/env python3
import argparse
import csv
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
from encoder.modules.lstm import SLSTM


DEFAULT_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
DEFAULT_CKPT = "/home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt"
DEFAULT_OUTDIR = "/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0304/wavtokenizer_featuremap_6wav/conv18_roles"

FILES = [
    {
        "id": "boy4_clean",
        "speaker": "boy4",
        "condition": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy4_clean_001.wav",
    },
    {
        "id": "boy7_clean",
        "speaker": "boy7",
        "condition": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean/nor_boy7_clean_001.wav",
    },
    {
        "id": "girl9_clean",
        "speaker": "girl9",
        "condition": "clean",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/clean_mix_test/recon/0224a/snr+10dB/nor_girl9_clean_001.wav",
    },
    {
        "id": "boy4_ldv",
        "speaker": "boy4",
        "condition": "ldv",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy4/box/nor_boy4_box_LDV_001.wav",
    },
    {
        "id": "boy7_ldv",
        "speaker": "boy7",
        "condition": "ldv",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/boy7/box/nor_boy7_box_LDV_001.wav",
    },
    {
        "id": "girl9_ldv",
        "speaker": "girl9",
        "condition": "ldv",
        "path": "/home/sbplab/ruizi/WavTokenize-feature-analysis/material/recon/0224a/girl9/box/nor_girl9_box_LDV_001.wav",
    },
]


@dataclass
class ConvLayerSpec:
    idx: int
    module_path: str
    human_name: str
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
                    human_name=f"L{idx:02d} top_sconv t{t_i}",
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
            specs.append(
                ConvLayerSpec(
                    idx=idx,
                    module_path=f"encoder.model[{t_i}].block[1]",
                    human_name=f"L{idx:02d} rb{t_i}_conv1",
                    stage="residual",
                    module=c1,
                )
            )
            idx += 1
            specs.append(
                ConvLayerSpec(
                    idx=idx,
                    module_path=f"encoder.model[{t_i}].block[3]",
                    human_name=f"L{idx:02d} rb{t_i}_conv2",
                    stage="residual",
                    module=c2,
                )
            )
            idx += 1
            if isinstance(sc, SConv1d):
                specs.append(
                    ConvLayerSpec(
                        idx=idx,
                        module_path=f"encoder.model[{t_i}].shortcut",
                        human_name=f"L{idx:02d} rb{t_i}_shortcut",
                        stage="residual",
                        module=sc,
                    )
                )
                idx += 1
    if len(specs) != 18:
        raise RuntimeError(f"Expected 18 conv-like layers but got {len(specs)}.")
    return specs


def top_level_desc(encoder: torch.nn.Module) -> List[str]:
    out = []
    for i, m in enumerate(encoder.model):
        if isinstance(m, SConv1d):
            s = m.conv.conv.stride[0]
            k = m.conv.conv.kernel_size[0]
            c0 = m.conv.conv.in_channels
            c1 = m.conv.conv.out_channels
            out.append(f"{i:02d}: SConv1d(k={k},s={s},{c0}->{c1})")
        elif isinstance(m, SEANetResnetBlock):
            out.append(f"{i:02d}: SEANetResnetBlock(2conv+shortcut)")
        elif isinstance(m, SLSTM):
            out.append(f"{i:02d}: SLSTM(num_layers={m.lstm.num_layers})")
        else:
            out.append(f"{i:02d}: {m.__class__.__name__}")
    return out


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


def make_layer_scalar_metrics(layer_out: torch.Tensor) -> Dict[str, float]:
    # layer_out: [C, T]
    abs_mean = float(layer_out.abs().mean().item())
    rms = float(torch.sqrt((layer_out ** 2).mean()).item())
    # Temporal detail proxy: channel-wise std over time, then mean channels.
    temp_std = float(layer_out.std(dim=-1, unbiased=False).mean().item())
    # Frame-to-frame delta proxy: first-order temporal difference.
    if layer_out.shape[-1] > 1:
        delta = layer_out[:, 1:] - layer_out[:, :-1]
        delta_rms = float(torch.sqrt((delta ** 2).mean()).item())
    else:
        delta_rms = 0.0
    return {
        "abs_mean": abs_mean,
        "rms": rms,
        "temp_std": temp_std,
        "delta_rms": delta_rms,
    }


def summarize_roles(
    conv_specs: List[ConvLayerSpec],
    per_file_vecs: Dict[str, Dict[int, torch.Tensor]],
    per_file_scalars: Dict[str, Dict[int, Dict[str, float]]],
) -> List[Dict]:
    role_rows = []

    clean_ids = [x["id"] for x in FILES if x["condition"] == "clean"]
    ldv_ids = [x["id"] for x in FILES if x["condition"] == "ldv"]
    same_spk_pairs = [("boy4_clean", "boy4_ldv"), ("boy7_clean", "boy7_ldv"), ("girl9_clean", "girl9_ldv")]
    clean_cross = [("boy4_clean", "boy7_clean"), ("boy4_clean", "girl9_clean"), ("boy7_clean", "girl9_clean")]
    ldv_cross = [("boy4_ldv", "boy7_ldv"), ("boy4_ldv", "girl9_ldv"), ("boy7_ldv", "girl9_ldv")]

    # For normalization
    temp_std_all = []
    for fid in per_file_scalars:
        for li in per_file_scalars[fid]:
            temp_std_all.append(per_file_scalars[fid][li]["temp_std"])
    tmin = float(min(temp_std_all))
    tmax = float(max(temp_std_all))
    tspan = max(tmax - tmin, 1e-8)

    for spec in conv_specs:
        li = spec.idx
        same_cos = np.mean([cos(per_file_vecs[a][li], per_file_vecs[b][li]) for a, b in same_spk_pairs]).item()
        clean_cross_cos = np.mean([cos(per_file_vecs[a][li], per_file_vecs[b][li]) for a, b in clean_cross]).item()
        ldv_cross_cos = np.mean([cos(per_file_vecs[a][li], per_file_vecs[b][li]) for a, b in ldv_cross]).item()
        cross_cos = float((clean_cross_cos + ldv_cross_cos) / 2.0)
        # Proxies
        noise_sensitivity = float(1.0 - same_cos)
        speaker_identity = float(max(0.0, same_cos - cross_cos))
        content_shared = float(max(0.0, cross_cos))
        temp_std_mean = float(np.mean([per_file_scalars[fid][li]["temp_std"] for fid in per_file_scalars]))
        temp_norm = float((temp_std_mean - tmin) / tspan)

        # Heuristic role label
        if noise_sensitivity >= 0.25 and temp_norm >= 0.50:
            role = "acoustic/noise-sensitive frontend"
        elif speaker_identity >= 0.12 and same_cos >= 0.70:
            role = "speaker/prosody-preserving"
        elif content_shared >= 0.80 and same_cos >= 0.80:
            role = "content/phonetic-shared"
        else:
            role = "mixed/transition"

        role_rows.append(
            {
                "layer_idx": li,
                "layer_name": spec.human_name,
                "module_path": spec.module_path,
                "stage": spec.stage,
                "same_speaker_clean_ldv_cos": float(same_cos),
                "cross_speaker_same_condition_cos": float(cross_cos),
                "noise_sensitivity": noise_sensitivity,
                "speaker_identity_score": speaker_identity,
                "content_shared_score": content_shared,
                "temporal_detail_score_norm": temp_norm,
                "role_proxy": role,
            }
        )
    return role_rows


def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_six_lines(
    outdir: Path,
    conv_specs: List[ConvLayerSpec],
    per_file_scalars: Dict[str, Dict[int, Dict[str, float]]],
    metric_key: str,
    ylabel: str,
    filename: str,
):
    x = list(range(len(conv_specs)))
    labels = [f"L{spec.idx:02d}" for spec in conv_specs]
    plt.figure(figsize=(14, 5))
    style_map = {
        "boy4_clean": ("#1f77b4", "-"),
        "boy7_clean": ("#2ca02c", "-"),
        "girl9_clean": ("#ff7f0e", "-"),
        "boy4_ldv": ("#1f77b4", "--"),
        "boy7_ldv": ("#2ca02c", "--"),
        "girl9_ldv": ("#ff7f0e", "--"),
    }
    for item in FILES:
        fid = item["id"]
        color, ls = style_map[fid]
        y = [per_file_scalars[fid][i][metric_key] for i in x]
        plt.plot(x, y, color=color, linestyle=ls, marker="o", markersize=3, linewidth=1.8, label=fid)

    plt.xticks(x, labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel(ylabel)
    plt.xlabel("18 conv-like layers")
    plt.title(f"Six-file layer trajectory ({metric_key})")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=220)
    plt.close()


def make_lora_recommendation(role_rows: List[Dict]) -> Dict:
    # Rank candidate layers to adapt for denoising:
    # high noise sensitivity + lower speaker-identity score
    scored = []
    for r in role_rows:
        score = 0.75 * r["noise_sensitivity"] + 0.25 * (1.0 - r["speaker_identity_score"])
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    adapt = [x[1] for x in scored[:6]]

    # Layers to freeze strongly:
    # strong speaker identity and strong content-shared
    freeze_rank = sorted(
        role_rows,
        key=lambda r: (r["speaker_identity_score"] + r["content_shared_score"]),
        reverse=True,
    )
    freeze = freeze_rank[:6]

    return {
        "adapt_top6": [
            {
                "layer_idx": r["layer_idx"],
                "layer_name": r["layer_name"],
                "module_path": r["module_path"],
                "reason": "high noise_sensitivity and lower speaker_identity_score",
            }
            for r in adapt
        ],
        "freeze_top6": [
            {
                "layer_idx": r["layer_idx"],
                "layer_name": r["layer_name"],
                "module_path": r["module_path"],
                "reason": "high speaker/content preservation score",
            }
            for r in freeze
        ],
    }


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

    per_file_raw: Dict[str, Dict[int, torch.Tensor]] = {}
    per_file_vecs: Dict[str, Dict[int, torch.Tensor]] = {}
    per_file_scalars: Dict[str, Dict[int, Dict[str, float]]] = {}

    for item in FILES:
        fid = item["id"]
        wav = read_audio_24k_mono(item["path"])
        raw = extract_layer_outputs(model, conv_specs, wav, device)
        per_file_raw[fid] = raw
        per_file_vecs[fid] = {}
        per_file_scalars[fid] = {}
        for li, out in raw.items():
            per_file_vecs[fid][li] = out.mean(dim=-1).float()  # [C]
            per_file_scalars[fid][li] = make_layer_scalar_metrics(out)

    role_rows = summarize_roles(conv_specs, per_file_vecs, per_file_scalars)
    lora_plan = make_lora_recommendation(role_rows)

    # Save six-line inputs
    six_line_rows = []
    for item in FILES:
        fid = item["id"]
        for li in range(18):
            row = {
                "file_id": fid,
                "speaker": item["speaker"],
                "condition": item["condition"],
                "layer_idx": li,
                "layer_name": conv_specs[li].human_name,
                "module_path": conv_specs[li].module_path,
                "stage": conv_specs[li].stage,
            }
            row.update(per_file_scalars[fid][li])
            six_line_rows.append(row)

    save_csv(
        outdir / "conv18_sixlines_metrics.csv",
        six_line_rows,
        [
            "file_id",
            "speaker",
            "condition",
            "layer_idx",
            "layer_name",
            "module_path",
            "stage",
            "abs_mean",
            "rms",
            "temp_std",
            "delta_rms",
        ],
    )

    save_csv(
        outdir / "conv18_role_metrics.csv",
        role_rows,
        [
            "layer_idx",
            "layer_name",
            "module_path",
            "stage",
            "same_speaker_clean_ldv_cos",
            "cross_speaker_same_condition_cos",
            "noise_sensitivity",
            "speaker_identity_score",
            "content_shared_score",
            "temporal_detail_score_norm",
            "role_proxy",
        ],
    )

    plot_six_lines(
        outdir=outdir,
        conv_specs=conv_specs,
        per_file_scalars=per_file_scalars,
        metric_key="temp_std",
        ylabel="Temporal Std (proxy for temporal detail / speech-rate dynamics)",
        filename="conv18_sixlines_temp_std.png",
    )
    plot_six_lines(
        outdir=outdir,
        conv_specs=conv_specs,
        per_file_scalars=per_file_scalars,
        metric_key="delta_rms",
        ylabel="Frame-delta RMS (proxy for fast-changing detail/noise sensitivity)",
        filename="conv18_sixlines_delta_rms.png",
    )

    summary = {
        "device": str(device),
        "config": args.config,
        "ckpt": args.ckpt,
        "top_level_encoder": top_level_desc(encoder),
        "conv18_layers": [
            {
                "layer_idx": s.idx,
                "layer_name": s.human_name,
                "module_path": s.module_path,
                "stage": s.stage,
            }
            for s in conv_specs
        ],
        "lora_recommendation": lora_plan,
        "note": (
            "Role labels are proxy-based from 6 files with same sentence. "
            "Useful for LoRA placement, not a strict causal proof of semantics."
        ),
    }
    (outdir / "conv18_roles_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    role_sorted_noise = sorted(role_rows, key=lambda r: r["noise_sensitivity"], reverse=True)
    role_sorted_speaker = sorted(role_rows, key=lambda r: r["speaker_identity_score"], reverse=True)
    role_sorted_content = sorted(role_rows, key=lambda r: r["content_shared_score"], reverse=True)

    md = []
    md.append("# Conv18 Layer Role Analysis for LoRA Denoising")
    md.append("")
    md.append("## What this means")
    md.append("- `noise_sensitivity`: same speaker clean/LDV 越不相似，值越高，表示該層受噪聲/條件影響更大。")
    md.append("- `speaker_identity_score`: 同一說話者跨條件相似度扣掉跨說話者相似度；越高表示越保留說話者特徵。")
    md.append("- `content_shared_score`: 跨說話者同條件相似度；越高表示更像共享語音內容/音素結構。")
    md.append("- `temp_std`/`delta_rms`: 時序變化強度，偏向節奏/細節敏感度代理。")
    md.append("")
    md.append("## Top noisy-sensitive layers (adapt candidate)")
    for r in role_sorted_noise[:8]:
        md.append(
            f"- L{r['layer_idx']:02d} {r['module_path']}: noise={r['noise_sensitivity']:.3f}, "
            f"speaker={r['speaker_identity_score']:.3f}, role={r['role_proxy']}"
        )
    md.append("")
    md.append("## Top speaker-preserving layers (prefer freeze)")
    for r in role_sorted_speaker[:8]:
        md.append(
            f"- L{r['layer_idx']:02d} {r['module_path']}: speaker={r['speaker_identity_score']:.3f}, "
            f"content={r['content_shared_score']:.3f}, role={r['role_proxy']}"
        )
    md.append("")
    md.append("## Top content-shared layers (prefer freeze)")
    for r in role_sorted_content[:8]:
        md.append(
            f"- L{r['layer_idx']:02d} {r['module_path']}: content={r['content_shared_score']:.3f}, "
            f"noise={r['noise_sensitivity']:.3f}, role={r['role_proxy']}"
        )
    md.append("")
    md.append("## LoRA proposal")
    md.append("- 建議只在 `adapt_top6` 加 LoRA（小 rank，如 r=4~8），其餘先凍結。")
    md.append("- `freeze_top6` 優先保持 frozen，避免破壞 WavTokenizer 原本模仿/內容保持能力。")
    md.append("- 若要更激進，可加第二輪：只額外解凍 1~2 個 mixed layer，觀察 token drift。")
    (outdir / "conv18_role_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] {outdir / 'conv18_sixlines_temp_std.png'}")
    print(f"[OK] {outdir / 'conv18_sixlines_delta_rms.png'}")
    print(f"[OK] {outdir / 'conv18_sixlines_metrics.csv'}")
    print(f"[OK] {outdir / 'conv18_role_metrics.csv'}")
    print(f"[OK] {outdir / 'conv18_roles_summary.json'}")
    print(f"[OK] {outdir / 'conv18_role_report.md'}")


if __name__ == "__main__":
    main()

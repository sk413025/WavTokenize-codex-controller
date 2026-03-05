#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decoder.pretrained import WavTokenizer
from encoder.modules.seanet import SEANetResnetBlock
from encoder.modules.conv import SConv1d
from encoder.modules.lstm import SLSTM


DEFAULT_CONFIG = "/home/sbplab/ruizi/WavTokenizer-main/configs/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
DEFAULT_CKPT = "/home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt"

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


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = (a.norm(p=2) * b.norm(p=2)).item()
    if denom == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def token_entropy(tokens_1d: torch.Tensor) -> float:
    vals, counts = torch.unique(tokens_1d, return_counts=True)
    probs = counts.float() / counts.sum().float()
    return float((-(probs * torch.log2(probs))).sum().item())


def describe_module(module: torch.nn.Module, idx: int) -> str:
    if isinstance(module, SConv1d):
        stride = module.conv.conv.stride[0]
        ksize = module.conv.conv.kernel_size[0]
        in_ch = module.conv.conv.in_channels
        out_ch = module.conv.conv.out_channels
        if stride > 1:
            return f"{idx:02d}: SConv1d downsample (k={ksize}, s={stride}, {in_ch}->{out_ch})"
        return f"{idx:02d}: SConv1d conv (k={ksize}, s={stride}, {in_ch}->{out_ch})"
    if isinstance(module, SEANetResnetBlock):
        return f"{idx:02d}: SEANetResnetBlock (2 conv in residual branch + 1 shortcut conv)"
    if isinstance(module, SLSTM):
        return f"{idx:02d}: SLSTM (num_layers={module.lstm.num_layers})"
    return f"{idx:02d}: {module.__class__.__name__}"


def collect_encoder_structure(encoder: torch.nn.Module) -> Dict:
    layers = list(encoder.model)
    structure = []
    conv_like_count = 0
    lstm_layer_count = 0
    for i, layer in enumerate(layers):
        item = {
            "index": i,
            "type": layer.__class__.__name__,
            "desc": describe_module(layer, i),
        }
        structure.append(item)
        if isinstance(layer, SConv1d):
            conv_like_count += 1
        elif isinstance(layer, SEANetResnetBlock):
            # residual branch conv2 + shortcut conv1 (true_skip=False in this model)
            conv_like_count += 3
        elif isinstance(layer, SLSTM):
            lstm_layer_count += layer.lstm.num_layers
    return {
        "top_level_module_count": len(layers),
        "conv_like_layer_count": conv_like_count,
        "lstm_internal_layer_count": lstm_layer_count,
        "layers": structure,
    }


def load_audio_24k_mono(path: str) -> torch.Tensor:
    wav_np, sr = sf.read(path, always_2d=True, dtype="float32")
    # soundfile gives [T, C], model expects [C, T]
    wav = torch.from_numpy(wav_np).transpose(0, 1)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def analyze_file(
    model: WavTokenizer,
    encoder: torch.nn.Module,
    path: str,
    device: torch.device,
) -> Dict:
    wav = load_audio_24k_mono(path)
    wav = wav.to(device)

    layer_maps: Dict[str, torch.Tensor] = {}
    handles = []
    layer_names: List[str] = []

    for i, layer in enumerate(encoder.model):
        name = f"{i:02d}_{layer.__class__.__name__}"
        layer_names.append(name)

        def _hook_factory(k: str):
            def _hook(_m, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                layer_maps[k] = out.detach()

            return _hook

        handles.append(layer.register_forward_hook(_hook_factory(name)))

    bw = torch.tensor([0], device=device)
    with torch.no_grad():
        features, codes = model.encode_infer(wav, bandwidth_id=bw)

    for h in handles:
        h.remove()

    # features: [B, C, T], codes: [K, B, T]
    feat = features.detach().squeeze(0).cpu()  # [C, T]
    codes = codes.detach().cpu()
    tokens = codes[0, 0].long()  # n_q=1

    layer_summary = {}
    layer_vectors = {}
    for name in layer_names:
        out = layer_maps[name].squeeze(0).detach().cpu()  # [C, T]
        pooled = out.mean(dim=-1)  # [C]
        layer_vectors[name] = pooled
        layer_summary[name] = {
            "channels": int(out.shape[0]),
            "frames": int(out.shape[1]),
            "abs_mean": float(out.abs().mean().item()),
            "std": float(out.std().item()),
        }

    counts = torch.bincount(tokens, minlength=4096)
    topk = torch.topk(counts, k=10)
    top_tokens = [
        {"token": int(tok), "count": int(cnt)}
        for tok, cnt in zip(topk.indices.tolist(), topk.values.tolist())
        if cnt > 0
    ]

    return {
        "path": path,
        "audio_samples": int(wav.shape[-1]),
        "duration_sec": float(wav.shape[-1] / 24000.0),
        "feature_shape": [int(feat.shape[0]), int(feat.shape[1])],
        "token_len": int(tokens.numel()),
        "token_unique": int(torch.unique(tokens).numel()),
        "token_entropy_bits": token_entropy(tokens),
        "token_top10": top_tokens,
        "feature_abs_mean": float(feat.abs().mean().item()),
        "feature_std": float(feat.std().item()),
        "feature_rms": float(torch.sqrt((feat ** 2).mean()).item()),
        "feature_timevar_mean": float(feat.var(dim=-1, unbiased=False).mean().item()),
        "feature_mean_vector": feat.mean(dim=-1),  # tensor
        "tokens_1d": tokens,  # tensor
        "features_ct": feat,  # tensor [C, T]
        "layer_vectors": layer_vectors,  # dict[str, tensor [C]]
        "layer_summary": layer_summary,
    }


def compare_pair(a: Dict, b: Dict, label: str) -> Dict:
    ta = a["tokens_1d"]
    tb = b["tokens_1d"]
    tf = min(int(ta.numel()), int(tb.numel()))
    token_match = float((ta[:tf] == tb[:tf]).float().mean().item())

    fa = a["features_ct"]
    fb = b["features_ct"]
    ff = min(fa.shape[-1], fb.shape[-1])
    fa_cut = fa[:, :ff]
    fb_cut = fb[:, :ff]
    frame_cos = F.cosine_similarity(fa_cut.t(), fb_cut.t(), dim=1).mean().item()
    mean_cos = cosine(a["feature_mean_vector"], b["feature_mean_vector"])

    layer_cos = {}
    for k in a["layer_vectors"].keys():
        layer_cos[k] = cosine(a["layer_vectors"][k], b["layer_vectors"][k])

    return {
        "pair": label,
        "token_match_ratio": token_match,
        "feature_mean_cosine": mean_cos,
        "feature_framewise_cosine_mean": float(frame_cos),
        "feature_std_abs_diff": abs(a["feature_std"] - b["feature_std"]),
        "token_entropy_abs_diff": abs(a["token_entropy_bits"] - b["token_entropy_bits"]),
        "layerwise_cosine": layer_cos,
    }


def pairwise_matrix(items: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    keys = list(items.keys())
    out = {}
    for i, ki in enumerate(keys):
        out[ki] = {}
        for j, kj in enumerate(keys):
            if i == j:
                out[ki][kj] = 1.0
                continue
            out[ki][kj] = cosine(items[ki]["feature_mean_vector"], items[kj]["feature_mean_vector"])
    return out


def trim_tensor_fields(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            continue
        if isinstance(v, dict):
            out[k] = trim_tensor_fields(v)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument(
        "--output_dir",
        default="/home/sbplab/ruizi/WavTokenize-feature-analysis/exp_0304/wavtokenizer_featuremap_6wav",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = WavTokenizer.from_pretrained0802(args.config, args.ckpt).to(device).eval()
    encoder = model.feature_extractor.encodec.encoder

    encoder_structure = collect_encoder_structure(encoder)

    results: Dict[str, Dict] = {}
    for item in FILES:
        results[item["id"]] = analyze_file(model, encoder, item["path"], device)
        results[item["id"]]["speaker"] = item["speaker"]
        results[item["id"]]["condition"] = item["condition"]

    pair_results = {
        "boy4_clean_vs_ldv": compare_pair(results["boy4_clean"], results["boy4_ldv"], "boy4 clean vs ldv"),
        "boy7_clean_vs_ldv": compare_pair(results["boy7_clean"], results["boy7_ldv"], "boy7 clean vs ldv"),
        "girl9_clean_vs_ldv": compare_pair(results["girl9_clean"], results["girl9_ldv"], "girl9 clean vs ldv"),
    }

    clean_items = {
        k: v for k, v in results.items() if v["condition"] == "clean"
    }
    ldv_items = {
        k: v for k, v in results.items() if v["condition"] == "ldv"
    }

    summary = {
        "device": str(device),
        "model_config": args.config,
        "model_ckpt": args.ckpt,
        "encoder_structure": encoder_structure,
        "per_file": {k: trim_tensor_fields(v) for k, v in results.items()},
        "clean_vs_ldv_pairs": pair_results,
        "speaker_similarity_matrix_feature_mean_cosine": {
            "clean": pairwise_matrix(clean_items),
            "ldv": pairwise_matrix(ldv_items),
            "all": pairwise_matrix(results),
        },
    }

    out_json = output_dir / "analysis_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Layerwise clean-vs-ldv table
    layer_table = {}
    for pair_name, pair_data in pair_results.items():
        layer_table[pair_name] = pair_data["layerwise_cosine"]
    (output_dir / "layerwise_clean_vs_ldv_cosine.json").write_text(
        json.dumps(layer_table, ensure_ascii=False, indent=2)
    )

    # Compact human-readable summary
    lines = []
    lines.append("WavTokenizer 6-file feature-map analysis")
    lines.append(f"Device: {device}")
    lines.append("")
    lines.append("Encoder structure:")
    lines.append(f"- top-level modules: {encoder_structure['top_level_module_count']}")
    lines.append(f"- conv-like layers (counting residual internal+shortcut conv): {encoder_structure['conv_like_layer_count']}")
    lines.append(f"- LSTM internal layers: {encoder_structure['lstm_internal_layer_count']}")
    lines.append("")
    lines.append("Per-file token/feature stats:")
    for k, v in summary["per_file"].items():
        lines.append(
            f"- {k}: token_len={v['token_len']}, token_unique={v['token_unique']}, "
            f"token_entropy={v['token_entropy_bits']:.3f}, feat_std={v['feature_std']:.5f}, "
            f"duration={v['duration_sec']:.2f}s"
        )
    lines.append("")
    lines.append("Same-speaker clean vs LDV:")
    for k, v in pair_results.items():
        lines.append(
            f"- {k}: token_match={v['token_match_ratio']:.4f}, "
            f"mean_feat_cos={v['feature_mean_cosine']:.4f}, frame_feat_cos={v['feature_framewise_cosine_mean']:.4f}"
        )
    (output_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n")

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {output_dir / 'analysis_summary.txt'}")
    print(f"[OK] Wrote {output_dir / 'layerwise_clean_vs_ldv_cosine.json'}")


if __name__ == "__main__":
    main()

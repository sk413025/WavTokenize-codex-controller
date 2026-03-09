"""
Attention Entropy Analysis

For specified checkpoint epochs, compute entropy of cross-attention weights
per token (normalized by log(K)) and summarize statistics. If the model does
not expose attention weights, the epoch is skipped.

Outputs under results_dir/analysis/attn_entropy/epoch_{E}/:
  - entropy_epoch_{E}.csv (rows: count, mean, std, p50, p90, peaked_frac@0.7)

Usage:
  python -u done/exp/analyze_attention_entropy.py \
    --results_dir results/crossattn_k4_... \
    --cache_dir /path/to/cache \
    --epochs 10 20 30 40 \
    --batch_size 32
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn
from model_zeroshot_crossattn_deep import ZeroShotDenoisingTransformerCrossAttnDeep
from model_zeroshot_crossattn_gated import ZeroShotDenoisingTransformerCrossAttnGated
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def load_model_with_attn(results_dir: Path, epoch: int, device: torch.device):
    ckpt_path = results_dir / f'checkpoint_epoch_{epoch}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    codebook = state['codebook']
    keys = list(state.keys())
    if any(k.startswith('cross_attn_fusion.gate') for k in keys):
        model = ZeroShotDenoisingTransformerCrossAttnGated(
            codebook=codebook, speaker_embed_dim=256, d_model=512, nhead=8,
            num_layers=4, dim_feedforward=2048, dropout=0.1, speaker_tokens=4
        ).to(device)
    elif any(k.startswith('fusion0.') or k.startswith('layers.') for k in keys):
        model = ZeroShotDenoisingTransformerCrossAttnDeep(
            codebook=codebook, speaker_embed_dim=256, d_model=512, nhead=8,
            num_layers=4, dim_feedforward=2048, dropout=0.1, speaker_tokens=4,
            inject_layers=(1,3)
        ).to(device)
    else:
        model = ZeroShotDenoisingTransformerCrossAttn(
            codebook=codebook, speaker_embed_dim=256, d_model=512, nhead=8,
            num_layers=4, dim_feedforward=2048, dropout=0.1, speaker_tokens=4
        ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def summarize(vals: torch.Tensor):
    # vals shape (N,)
    if vals.numel() == 0:
        return 0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    v = vals.view(-1).float().cpu()
    count = v.numel()
    mean = v.mean().item()
    std = v.std(unbiased=False).item()
    p50 = v.kthvalue(max(1, int(0.5*count))).values.item() if count>0 else float('nan')
    p90 = v.kthvalue(max(1, int(0.9*count))).values.item() if count>0 else float('nan')
    peaked = (v > 0.7).float().mean().item()  # peaked fraction (normalized entropy < 0.3)
    return count, mean, std, p50, p90, peaked


def analyze_epoch(results_dir: Path, cache_dir: Path, epoch: int, batch_size: int = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_with_attn(results_dir, epoch, device)

    ds = ZeroShotAudioDatasetCached(str(cache_dir / 'val_cache.pt'))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=cached_collate_fn, num_workers=0)

    entropies = []
    with torch.no_grad():
        for batch in dl:
            noisy = batch['noisy_tokens'].to(device)
            spk = batch['speaker_embeddings'].to(device)
            try:
                out = model(noisy, spk, return_logits=True, return_attention=True)
            except TypeError:
                # model may not support returning attention; skip
                return False
            if isinstance(out, tuple) and len(out) >= 2:
                logits = out[0]
                attn_w = out[1]  # (B,T,K)
            else:
                return False
            B,T,K = attn_w.shape
            p = attn_w.clamp_min(1e-12)
            h = -(p * p.log()).sum(dim=-1)  # (B,T)
            h_max = math.log(K)
            h_norm = (h / max(1e-12, h_max)).clamp(0,1)  # normalized entropy
            entropies.append(h_norm.view(-1).cpu())

    if not entropies:
        return False
    ent = torch.cat(entropies, dim=0)
    count, mean, std, p50, p90, peaked = summarize(ent)

    out_dir = results_dir / 'analysis' / 'attn_entropy' / f'epoch_{epoch}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'entropy_epoch_{epoch}.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['count','mean','std','p50','p90','peaked_frac_gt0.7'])
        writer.writerow([count, f"{mean:.6f}", f"{std:.6f}", f"{p50:.6f}", f"{p90:.6f}", f"{peaked:.6f}"])
    print(f"✓ Attention entropy saved: {out_csv}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir = Path(args.cache_dir)
    for e in args.epochs:
        analyze_epoch(results_dir, cache_dir, e, args.batch_size)

if __name__ == '__main__':
    main()


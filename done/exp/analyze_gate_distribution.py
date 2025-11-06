"""
Gate Distribution vs Decision Margin (Gated models only)

For specified checkpoint epochs, compute per-token decision margin under
normal speaker and summarize gate value distribution per margin bin.

Outputs under results_dir/analysis/gate_distribution/epoch_{E}/:
  - gate_stats_epoch_{E}.csv

Columns:
  bin_left, bin_right, count, gate_mean, gate_std, gate_p10, gate_p50, gate_p90,
  low_gate_frac(<0.2), high_gate_frac(>0.8)

Usage:
  python -u done/exp/analyze_gate_distribution.py \
    --results_dir results/crossattn_k4_gate_... \
    --cache_dir /path/to/cache \
    --epochs 10 20 30 40 \
    --batch_size 32
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_zeroshot_crossattn_gated import ZeroShotDenoisingTransformerCrossAttnGated
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def load_gated(results_dir: Path, epoch: int, device: torch.device):
    ckpt_path = results_dir / f'checkpoint_epoch_{epoch}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    codebook = state['codebook']
    # quick check for gate key
    if not any(k.startswith('cross_attn_fusion.gate') for k in state.keys()):
        return None
    model = ZeroShotDenoisingTransformerCrossAttnGated(
        codebook=codebook, speaker_embed_dim=256, d_model=512, nhead=8,
        num_layers=4, dim_feedforward=2048, dropout=0.1, speaker_tokens=4
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def analyze_epoch(results_dir: Path, cache_dir: Path, epoch: int, batch_size: int = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_gated(results_dir, epoch, device)
    if model is None:
        print(f"skip (not gated): {results_dir} epoch {epoch}")
        return False

    ds = ZeroShotAudioDatasetCached(str(cache_dir / 'val_cache.pt'))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=cached_collate_fn, num_workers=0)

    bins = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    nb = len(bins)-1
    def bindex(m):
        for i in range(nb):
            if bins[i] <= m < bins[i+1]:
                return i
        return nb-1
    stats = {
        'count':[0]*nb,
        'gates': [ [] for _ in range(nb) ],
        'low_gate':[0]*nb,
        'high_gate':[0]*nb,
    }

    with torch.no_grad():
        for batch in dl:
            noisy = batch['noisy_tokens'].to(device)
            clean = batch['clean_tokens'].to(device)
            spk = batch['speaker_embeddings'].to(device)
            mask = (clean != 0)
            logits, attn_w, gate = model(noisy, spk, return_logits=True, return_attention=True)
            probs = torch.softmax(logits, dim=-1)
            p_top, idx_top = probs.topk(k=2, dim=-1)
            margin = (p_top[...,0] - p_top[...,1]).clamp(0,1)
            B,T = margin.shape
            for b in range(B):
                for t in range(T):
                    if not bool(mask[b,t].item()):
                        continue
                    bi = bindex(float(margin[b,t].item()))
                    g = float(gate[b,t,0].item())
                    stats['count'][bi] += 1
                    stats['gates'][bi].append(g)
                    if g < 0.2: stats['low_gate'][bi] += 1
                    if g > 0.8: stats['high_gate'][bi] += 1

    out_dir = results_dir / 'analysis' / 'gate_distribution' / f'epoch_{epoch}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'gate_stats_epoch_{epoch}.csv'
    with open(out_csv,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['bin_left','bin_right','count','gate_mean','gate_std','gate_p10','gate_p50','gate_p90','low_gate_frac','high_gate_frac'])
        import numpy as np
        for i in range(nb):
            g = np.array(stats['gates'][i], dtype=float)
            cnt = max(1, stats['count'][i])
            if g.size == 0:
                mean=std=p10=p50=p90=0.0
            else:
                mean = float(g.mean()); std = float(g.std())
                p10 = float(np.percentile(g,10)); p50=float(np.percentile(g,50)); p90=float(np.percentile(g,90))
            low = stats['low_gate'][i]/cnt
            high = stats['high_gate'][i]/cnt
            w.writerow([bins[i], bins[i+1], stats['count'][i], f"{mean:.6f}", f"{std:.6f}", f"{p10:.6f}", f"{p50:.6f}", f"{p90:.6f}", f"{low:.6f}", f"{high:.6f}"])
    print(f"✓ Gate distribution saved: {out_csv}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    rd = Path(args.results_dir); cd = Path(args.cache_dir)
    for e in args.epochs:
        analyze_epoch(rd, cd, e, args.batch_size)

if __name__ == '__main__':
    main()


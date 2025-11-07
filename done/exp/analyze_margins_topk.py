"""
Decision Margin & Top-k Stability Analysis

For specified checkpoint epochs, compute per-token decision margin under
normal speaker, then evaluate flip rates, delta accuracy, and top-k overlap
when speaker is zeroed or randomized.

Outputs under results_dir/analysis/margins_topk/epoch_{E}/:
  - margins_bins_epoch_{E}.csv
  - flip_rate_epoch_{E}.png
  - delta_acc_epoch_{E}.png
  - topk_overlap_epoch_{E}.png

Usage:
  python -u done/exp/analyze_margins_topk.py \
    --results_dir results/crossattn_k4_... \
    --cache_dir /path/to/cache \
    --epochs 10 20 30 40 \
    --k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn
from model_zeroshot_crossattn_deep import ZeroShotDenoisingTransformerCrossAttnDeep
from model_zeroshot_crossattn_gated import ZeroShotDenoisingTransformerCrossAttnGated
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def _infer_num_layers_from_state(state: dict, default_layers: int = 4) -> int:
    max_idx = -1
    for k in state.keys():
        if k.startswith('transformer_encoder.layers.'):
            parts = k.split('.')
            if len(parts) > 2:
                try:
                    idx = int(parts[2])
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    pass
    return (max_idx + 1) if max_idx >= 0 else default_layers


def load_model(results_dir: Path, epoch: int, device: torch.device):
    ckpt_path = results_dir / f'checkpoint_epoch_{epoch}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    codebook = state['codebook']
    keys = list(state.keys())
    inferred_layers = _infer_num_layers_from_state(state, default_layers=4)
    if any(k.startswith('fusion0.') or k.startswith('layers.') for k in keys):
        model = ZeroShotDenoisingTransformerCrossAttnDeep(
            codebook=codebook,
            speaker_embed_dim=256,
            d_model=512,
            nhead=8,
            num_layers=inferred_layers,
            dim_feedforward=2048,
            dropout=0.1,
            speaker_tokens=4,
            inject_layers=(1,3),
        ).to(device)
    elif any(k.startswith('cross_attn_fusion.gate') for k in keys):
        model = ZeroShotDenoisingTransformerCrossAttnGated(
            codebook=codebook,
            speaker_embed_dim=256,
            d_model=512,
            nhead=8,
            num_layers=inferred_layers,
            dim_feedforward=2048,
            dropout=0.1,
            speaker_tokens=4,
        ).to(device)
    else:
        model = ZeroShotDenoisingTransformerCrossAttn(
            codebook=codebook,
            speaker_embed_dim=256,
            d_model=512,
            nhead=8,
            num_layers=inferred_layers,
            dim_feedforward=2048,
            dropout=0.1,
            speaker_tokens=4,
        ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def analyze_epoch(results_dir: Path, cache_dir: Path, epoch: int, k: int = 5, batch_size: int = 64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(results_dir, epoch, device)

    ds = ZeroShotAudioDatasetCached(str(cache_dir / 'val_cache.pt'))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=cached_collate_fn, num_workers=0)

    # Fixed margin bins
    bins = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 1.01]
    nb = len(bins) - 1
    def bin_index(m):
        # m in [0,1]
        for i in range(nb):
            if bins[i] <= m < bins[i+1]:
                return i
        return nb-1

    stats = {
        'zero': {'count':[0]*nb, 'flips':[0]*nb, 'acc_norm':[0]*nb, 'acc_var':[0]*nb, 'topk_overlap_sum':[0.0]*nb, 'target_in_topk_norm':[0]*nb, 'target_in_topk_var':[0]*nb},
        'random': {'count':[0]*nb, 'flips':[0]*nb, 'acc_norm':[0]*nb, 'acc_var':[0]*nb, 'topk_overlap_sum':[0.0]*nb, 'target_in_topk_norm':[0]*nb, 'target_in_topk_var':[0]*nb},
    }

    with torch.no_grad():
        for batch in dl:
            noisy = batch['noisy_tokens'].to(device)
            clean = batch['clean_tokens'].to(device)
            spk = batch['speaker_embeddings'].to(device)
            mask = (clean != 0)

            logits_norm = model(noisy, spk, return_logits=True)
            probs_norm = torch.softmax(logits_norm, dim=-1)
            p_top, idx_top = probs_norm.topk(k=max(k,2), dim=-1)  # (B,T,k)
            p1 = p_top[...,0]
            p2 = p_top[...,1]
            pred_norm = idx_top[...,0]
            margin = (p1 - p2).clamp(min=0.0, max=1.0)

            # Variants
            logits_zero = model(noisy, torch.zeros_like(spk), return_logits=True)
            probs_zero = torch.softmax(logits_zero, dim=-1)
            pred_zero = probs_zero.argmax(dim=-1)

            logits_rand = model(noisy, torch.randn_like(spk), return_logits=True)
            probs_rand = torch.softmax(logits_rand, dim=-1)
            pred_rand = probs_rand.argmax(dim=-1)

            # Top-k sets and overlaps
            topk_norm = idx_top[...,:k]
            topk_zero = probs_zero.topk(k, dim=-1).indices
            topk_rand = probs_rand.topk(k, dim=-1).indices

            for name, pred_var, topk_var in [('zero', pred_zero, topk_zero), ('random', pred_rand, topk_rand)]:
                # Flatten per token
                B,T = pred_var.shape
                for b in range(B):
                    for t in range(T):
                        if not bool(mask[b,t].item()):
                            continue
                        m = float(margin[b,t].item())
                        bi = bin_index(m)
                        stats[name]['count'][bi] += 1
                        # flip
                        if int(pred_var[b,t].item()) != int(pred_norm[b,t].item()):
                            stats[name]['flips'][bi] += 1
                        # acc
                        if int(pred_norm[b,t].item()) == int(clean[b,t].item()):
                            stats[name]['acc_norm'][bi] += 1
                        if int(pred_var[b,t].item()) == int(clean[b,t].item()):
                            stats[name]['acc_var'][bi] += 1
                        # top-k overlap
                        set_n = set(topk_norm[b,t].tolist())
                        set_v = set(topk_var[b,t].tolist())
                        overlap = len(set_n & set_v)/float(k)
                        stats[name]['topk_overlap_sum'][bi] += overlap
                        # target-in-topk
                        target = int(clean[b,t].item())
                        if target in set_n:
                            stats[name]['target_in_topk_norm'][bi] += 1
                        if target in set_v:
                            stats[name]['target_in_topk_var'][bi] += 1

    # Write CSV and plots
    out_dir = results_dir / 'analysis' / 'margins_topk' / f'epoch_{epoch}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'margins_bins_epoch_{epoch}.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_left','bin_right','condition','count','flip_rate','acc_norm_pct','acc_var_pct','delta_acc_pct','topk_overlap_mean','target_in_topk_norm_pct','target_in_topk_var_pct','delta_target_in_topk_pct'])
        for cond in ['zero','random']:
            for i in range(nb):
                cnt = max(1, stats[cond]['count'][i])
                flip_rate = stats[cond]['flips'][i]/cnt*100
                accn = stats[cond]['acc_norm'][i]/cnt*100
                accv = stats[cond]['acc_var'][i]/cnt*100
                delta_acc = accv - accn
                topk_overlap_mean = stats[cond]['topk_overlap_sum'][i]/cnt*100
                tin_n = stats[cond]['target_in_topk_norm'][i]/cnt*100
                tin_v = stats[cond]['target_in_topk_var'][i]/cnt*100
                dtin = tin_v - tin_n
                writer.writerow([bins[i], bins[i+1], cond, stats[cond]['count'][i], f"{flip_rate:.4f}", f"{accn:.4f}", f"{accv:.4f}", f"{delta_acc:.4f}", f"{topk_overlap_mean:.4f}", f"{tin_n:.4f}", f"{tin_v:.4f}", f"{dtin:.4f}"])

    # Plots
    import pandas as pd
    df = pd.read_csv(out_csv)
    def plot_metric(metric, ylabel, filename):
        plt.figure(figsize=(10,4))
        for cond, color in [('zero','tab:blue'),('random','tab:orange')]:
            sub = df[df['condition']==cond]
            # x axis as bin centers
            xs = [(l+r)/2 for l,r in zip(sub['bin_left'], sub['bin_right'])]
            ys = sub[metric].astype(float)
            plt.plot(xs, ys, marker='o', label=cond, color=color)
        plt.xlabel('margin (p1 - p2)')
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200, bbox_inches='tight')
        plt.close()

    plot_metric('flip_rate', 'Flip Rate (%)', f'flip_rate_epoch_{epoch}.png')
    plot_metric('delta_acc_pct', 'Delta Acc (variant - normal) pp', f'delta_acc_epoch_{epoch}.png')
    plot_metric('topk_overlap_mean', 'Top-k Overlap (%)', f'topk_overlap_epoch_{epoch}.png')

    print(f"✓ Margins & Top-k saved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir = Path(args.cache_dir)
    for e in args.epochs:
        analyze_epoch(results_dir, cache_dir, e, k=args.k, batch_size=args.batch_size)


if __name__ == '__main__':
    main()

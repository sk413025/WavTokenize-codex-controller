"""
Influence Breakdown Analysis (C->W / W->C / W->W / C->C)

For each specified checkpoint epoch, compute how predictions change when
speaker embedding is zeroed or randomized, relative to normal speaker.

Outputs per epoch under results_dir/analysis/influence_breakdown/epoch_{E}/:
  - breakdown_epoch_{E}.csv (rows per condition: normal->zero, normal->random)

Columns:
  condition, total_tokens, valid_tokens, acc_normal, acc_variant,
  c2c, c2w, w2c, w2w, c2c_pct, c2w_pct, w2c_pct, w2w_pct, net_acc_delta

Usage:
  python -u done/exp/analyze_influence_breakdown.py \
    --results_dir results/crossattn_k4_100epochs_... \
    --cache_dir /path/to/cache_dir \
    --epochs 10 20 30 40
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
from typing import List

import torch
from torch.utils.data import DataLoader

from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn
from model_zeroshot_crossattn_deep import ZeroShotDenoisingTransformerCrossAttnDeep
from model_zeroshot_crossattn_gated import ZeroShotDenoisingTransformerCrossAttnGated
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def _infer_num_layers_from_state(state: dict, default_layers: int = 4) -> int:
    """Infer transformer encoder depth from checkpoint keys."""
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
    # auto-detect deep injection checkpoints
    keys = list(state.keys())
    inferred_layers = _infer_num_layers_from_state(state, default_layers=4)
    # Deep-injection checkpoints
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
    # Gated checkpoints
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


def analyze_epoch(results_dir: Path, cache_dir: Path, epoch: int, batch_size: int = 64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(results_dir, epoch, device)

    ds = ZeroShotAudioDatasetCached(str(cache_dir / 'val_cache.pt'))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=cached_collate_fn, num_workers=0)

    totals = {
        'zero': {'total': 0, 'valid': 0, 'acc_norm': 0, 'acc_var': 0, 'c2c': 0, 'c2w': 0, 'w2c': 0, 'w2w': 0},
        'random': {'total': 0, 'valid': 0, 'acc_norm': 0, 'acc_var': 0, 'c2c': 0, 'c2w': 0, 'w2c': 0, 'w2w': 0},
    }

    with torch.no_grad():
        for batch in dl:
            noisy = batch['noisy_tokens'].to(device)
            clean = batch['clean_tokens'].to(device)
            spk = batch['speaker_embeddings'].to(device)
            mask = (clean != 0)

            logits_norm = model(noisy, spk, return_logits=True)
            pred_norm = logits_norm.argmax(dim=-1)

            # Zero speaker
            logits_zero = model(noisy, torch.zeros_like(spk), return_logits=True)
            pred_zero = logits_zero.argmax(dim=-1)

            # Random speaker
            logits_rand = model(noisy, torch.randn_like(spk), return_logits=True)
            pred_rand = logits_rand.argmax(dim=-1)

            for name, pred_var in [('zero', pred_zero), ('random', pred_rand)]:
                totals[name]['total'] += mask.numel()
                valid = mask.sum().item()
                totals[name]['valid'] += valid
                acc_norm = ((pred_norm == clean) & mask).sum().item()
                acc_var = ((pred_var == clean) & mask).sum().item()
                totals[name]['acc_norm'] += acc_norm
                totals[name]['acc_var'] += acc_var

                # Breakdown relative to normal
                norm_correct = (pred_norm == clean)
                var_correct = (pred_var == clean)
                c2c = (norm_correct & var_correct & mask).sum().item()
                c2w = (norm_correct & (~var_correct) & mask).sum().item()
                w2c = ((~norm_correct) & var_correct & mask).sum().item()
                w2w = ((~norm_correct) & (~var_correct) & mask).sum().item()
                totals[name]['c2c'] += c2c
                totals[name]['c2w'] += c2w
                totals[name]['w2c'] += w2c
                totals[name]['w2w'] += w2w

    # Write CSV
    out_dir = results_dir / 'analysis' / 'influence_breakdown' / f'epoch_{epoch}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'breakdown_epoch_{epoch}.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition','total_tokens','valid_tokens','acc_normal','acc_variant','acc_normal_pct','acc_variant_pct','c2c','c2w','w2c','w2w','c2c_pct','c2w_pct','w2c_pct','w2w_pct','net_acc_delta_pct'])
        for cond in ['zero','random']:
            t = totals[cond]
            v = max(1, t['valid'])
            accn = t['acc_norm']/v*100
            accv = t['acc_var']/v*100
            c2c_pct = t['c2c']/v*100
            c2w_pct = t['c2w']/v*100
            w2c_pct = t['w2c']/v*100
            w2w_pct = t['w2w']/v*100
            writer.writerow([cond, t['total'], t['valid'], t['acc_norm'], t['acc_var'], f"{accn:.4f}", f"{accv:.4f}", t['c2c'], t['c2w'], t['w2c'], t['w2w'], f"{c2c_pct:.4f}", f"{c2w_pct:.4f}", f"{w2c_pct:.4f}", f"{w2w_pct:.4f}", f"{(accv-accn):.4f}"])

    print(f"✓ Influence breakdown saved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir = Path(args.cache_dir)

    for e in args.epochs:
        analyze_epoch(results_dir, cache_dir, e, args.batch_size)


if __name__ == '__main__':
    main()

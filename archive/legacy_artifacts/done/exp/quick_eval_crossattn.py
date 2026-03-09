"""
Quick Eval for Cross-Attention (K>1)

Given a results dir (with best_model.pth or checkpoint_epoch_*.pth) and
./data/{val_cache.pt}, compute:
  - Attention weight stats (mean/std/min/max, token-wise variance)
  - Speaker influence (prediction change vs zero/random speaker)

Usage:
  python -u done/exp/quick_eval_crossattn.py --results_dir <RESULTS_DIR>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def pick_checkpoint(results_dir: Path, epoch: int | None = None) -> Path:
    best = results_dir / 'best_model.pth'
    if epoch is not None:
        ckpt = results_dir / f'checkpoint_epoch_{epoch}.pth'
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint_epoch_{epoch}.pth not found in {results_dir}")
        return ckpt
    if best.exists():
        return best
    ckpts = sorted(results_dir.glob('checkpoint_epoch_*.pth'), key=lambda p: int(p.stem.split('_')[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {results_dir}")
    return ckpts[-1]


def build_model(ckpt_path: Path, device: torch.device) -> ZeroShotDenoisingTransformerCrossAttn:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    codebook = state['codebook']
    # Infer hyperparams
    d_model = codebook.shape[1]
    vocab = codebook.shape[0]
    # Try detecting speaker_tokens from state
    speaker_tokens = 4
    for k in state.keys():
        if 'cross_attn_fusion.spk_pos' in k:
            speaker_tokens = state[k].shape[1]
            break

    model = ZeroShotDenoisingTransformerCrossAttn(
        codebook=codebook,
        speaker_embed_dim=256,
        d_model=d_model,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        speaker_tokens=speaker_tokens,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def attention_stats(model, batch, device) -> Dict[str, Any]:
    noisy = batch['noisy_tokens'].to(device)
    spk = batch['speaker_embeddings'].to(device)
    logits, attn = model(noisy, spk, return_logits=True, return_attention=True)
    stats = {
        'mean': float(attn.mean().item()),
        'std': float(attn.std().item()),
        'min': float(attn.min().item()),
        'max': float(attn.max().item()),
        'var_across_tokens_mean': float(attn.var(dim=1).mean().item()),
        'shape': tuple(attn.shape),
    }
    return stats


@torch.no_grad()
def speaker_influence(model, batch, device) -> Dict[str, Any]:
    noisy = batch['noisy_tokens'].to(device)
    clean = batch['clean_tokens'].to(device)
    spk = batch['speaker_embeddings'].to(device)

    logits_norm = model(noisy, spk, return_logits=True)
    pred_norm = logits_norm.argmax(dim=-1)

    logits_zero = model(noisy, torch.zeros_like(spk), return_logits=True)
    pred_zero = logits_zero.argmax(dim=-1)

    logits_rand = model(noisy, torch.randn_like(spk), return_logits=True)
    pred_rand = logits_rand.argmax(dim=-1)

    total = pred_norm.numel()
    change_zero = (pred_norm != pred_zero).sum().item()/total
    change_rand = (pred_norm != pred_rand).sum().item()/total

    acc_norm = (pred_norm == clean).float().mean().item()
    acc_zero = (pred_zero == clean).float().mean().item()
    acc_rand = (pred_rand == clean).float().mean().item()

    return {
        'change_zero': change_zero,
        'change_random': change_rand,
        'acc_norm': acc_norm,
        'acc_zero': acc_zero,
        'acc_rand': acc_rand,
        'acc_drop_zero': acc_norm - acc_zero,
        'acc_drop_random': acc_norm - acc_rand,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epoch', type=int, default=None, help='Use checkpoint_epoch_{epoch}.pth instead of best_model')
    ap.add_argument('--cache_dir', type=str, default='./data', help='Directory containing train_cache.pt/val_cache.pt')
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = pick_checkpoint(results_dir, epoch=args.epoch)
    print(f"Using checkpoint: {ckpt_path.name}")

    model = build_model(ckpt_path, device)

    val_path = Path(args.cache_dir) / 'val_cache.pt'
    if not val_path.exists():
        raise FileNotFoundError("./data/val_cache.pt not found. Generate cache first.")

    ds = ZeroShotAudioDatasetCached(str(val_path))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=cached_collate_fn, num_workers=0)
    batch = next(iter(dl))

    attn = attention_stats(model, batch, device)
    infl = speaker_influence(model, batch, device)

    print("\nAttention stats:")
    for k, v in attn.items():
        print(f"  {k}: {v}")

    print("\nSpeaker influence:")
    for k, v in infl.items():
        if 'acc' in k:
            print(f"  {k}: {v*100:.2f}%")
        else:
            print(f"  {k}: {v*100:.2f}% tokens changed")


if __name__ == '__main__':
    main()

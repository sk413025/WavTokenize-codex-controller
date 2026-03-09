"""
Smoke Test: Cross-Attention with K=4 speaker tokens

- Trains the updated model for 3 epochs on a tiny synthetic dataset
  (no external cache needed)
- Reports quick metrics:
  * Attention weights stats (mean/std/min/max, variance across tokens)
  * Speaker influence: prediction change rate vs zero/random speaker

This is for sanity-checking the K>1 change eliminates S=1 degeneracy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn


@dataclass
class SynthConfig:
    vocab_size: int = 4096
    d_model: int = 512
    speaker_dim: int = 256
    seq_len: int = 120
    num_samples: int = 256
    batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-4
    speaker_tokens: int = 4
    seed: int = 1337


class SynthDataset(Dataset):
    def __init__(self, cfg: SynthConfig):
        g = torch.Generator().manual_seed(cfg.seed)
        self.noisy_tokens = torch.randint(0, cfg.vocab_size, (cfg.num_samples, cfg.seq_len), generator=g)
        # Identity mapping as target (denoising not necessary for smoke test)
        self.clean_tokens = self.noisy_tokens.clone()
        self.speaker_embeddings = torch.randn(cfg.num_samples, cfg.speaker_dim, generator=g)

    def __len__(self):
        return self.noisy_tokens.size(0)

    def __getitem__(self, idx):
        return {
            'noisy_tokens': self.noisy_tokens[idx],
            'clean_tokens': self.clean_tokens[idx],
            'speaker_embeddings': self.speaker_embeddings[idx],
        }


def collate_fn(batch):
    noisy = torch.stack([b['noisy_tokens'] for b in batch])
    clean = torch.stack([b['clean_tokens'] for b in batch])
    spk = torch.stack([b['speaker_embeddings'] for b in batch])
    return {'noisy_tokens': noisy, 'clean_tokens': clean, 'speaker_embeddings': spk}


def train_one_epoch(model, dl, opt, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for batch in dl:
        noisy = batch['noisy_tokens'].to(device)
        clean = batch['clean_tokens'].to(device)
        spk = batch['speaker_embeddings'].to(device)

        logits = model(noisy, spk, return_logits=True)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), clean.reshape(B*T))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        total_correct += (pred == clean).sum().item()
        total_tokens += B*T

    return total_loss/len(dl), total_correct/total_tokens


@torch.no_grad()
def attention_stats(model, batch, device) -> Dict[str, Any]:
    noisy = batch['noisy_tokens'].to(device)
    spk = batch['speaker_embeddings'].to(device)
    logits, attn = model(noisy, spk, return_logits=True, return_attention=True)
    # attn: (B, T, K)
    stats = {
        'mean': float(attn.mean().item()),
        'std': float(attn.std().item()),
        'min': float(attn.min().item()),
        'max': float(attn.max().item()),
        'var_across_tokens_mean': float(attn.var(dim=1).mean().item()),  # token-wise variance averaged
    }
    return stats


@torch.no_grad()
def speaker_influence(model, batch, device) -> Dict[str, Any]:
    noisy = batch['noisy_tokens'].to(device)
    clean = batch['clean_tokens'].to(device)
    spk = batch['speaker_embeddings'].to(device)

    # Normal
    logits_norm = model(noisy, spk, return_logits=True)
    pred_norm = logits_norm.argmax(dim=-1)

    # Zero
    logits_zero = model(noisy, torch.zeros_like(spk), return_logits=True)
    pred_zero = logits_zero.argmax(dim=-1)

    # Random
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
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--speaker_tokens', type=int, default=4)
    cfg_args = p.parse_args()

    cfg = SynthConfig(epochs=cfg_args.epochs, batch_size=cfg_args.batch_size, speaker_tokens=cfg_args.speaker_tokens)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    print(f"Device: {device}")

    # Model with random codebook for speed
    codebook = torch.randn(cfg.vocab_size, cfg.d_model)
    model = ZeroShotDenoisingTransformerCrossAttn(
        codebook=codebook,
        speaker_embed_dim=cfg.speaker_dim,
        d_model=cfg.d_model,
        nhead=8,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        speaker_tokens=cfg.speaker_tokens,
    ).to(device)

    train_ds = SynthDataset(cfg)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    print("Training...")
    for e in range(1, cfg.epochs+1):
        loss, acc = train_one_epoch(model, train_dl, opt, criterion, device)
        print(f"  Epoch {e}/{cfg.epochs} - loss={loss:.4f}, acc={acc*100:.2f}%")

    # Metrics on a fixed batch
    fixed_batch = next(iter(train_dl))
    attn = attention_stats(model, fixed_batch, device)
    infl = speaker_influence(model, fixed_batch, device)

    print("\nAttention stats (K=4):")
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


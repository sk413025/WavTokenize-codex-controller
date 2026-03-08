#!/usr/bin/env python3
"""
exp_0305c: 官方 WavTokenizer 起點 + tail_lock anchor + 改良 LR schedule + 動態 lambda

改良重點（對比 exp_0305b expA/expB 的問題）：

問題：
  - expA/expB 的 best val_mse 都在 epoch 10-12 出現，之後震盪退步
  - 原因1：LR schedule 太激進（5ep warmup → cosine decay → epoch 50 後 LR < 6e-5）
  - 原因2：anchor lambda 固定，expA λ=1.0 太弱無法壓制偏移，expB λ=3.0 太強擠壓降噪空間

解法：
  1. 三段式 LR schedule:
       Epoch 1-10:    warmup   1e-5 → 1e-4
       Epoch 11-150:  constant 1e-4
       Epoch 151-300: cosine   1e-4 → 1e-6

  2. 動態 lambda_anchor:
       Epoch 1-50:    λ = 0.5   (先讓降噪自由學)
       Epoch 51-150:  λ = 1.5   (逐步加強約束)
       Epoch 151-300: λ = 3.0   (收斂期強力鎖定)

  錨定層：tail_lock (L16, L17)，因 expB 顯示前層約束改善有限

架構（與 exp_0305b 完全相同）：
  Encoder LoRA (全 18 層, rank=64, trainable)
  Decoder (frozen, 官方 WavTokenizer)
  起點：官方 WavTokenizer (NO exp_0217/0224a 先驗)

存檔規則：
  - 每 25 epoch：loss 曲線圖 + train/val 音檔
  - 每 25 epoch：checkpoint
  - best_model.pt：val_mse 最佳
"""

import argparse
import atexit
import json
import math
import gc
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0216.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum
from exp_0224.models_no_vq import TeacherStudentNoVQ

SAMPLE_RATE = 24000

# ── 沿用 exp_0305b 的 Loss 函數 ──────────────────────────────────────────────
try:
    from exp_0305b.train_0224a_anchor import (
        MultiResolutionSTFTLoss,
        MelReconstructionLoss,
        LayerHookBank,
        build_conv18_modules,
        compute_anchor_loss,
        preset_to_layers,
        parse_layer_ids,
        parse_layer_weights,
    )
except ImportError:
    # fallback: 直接 import from parent path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'exp_0305b'))
    from train_0224a_anchor import (
        MultiResolutionSTFTLoss,
        MelReconstructionLoss,
        LayerHookBank,
        build_conv18_modules,
        compute_anchor_loss,
        preset_to_layers,
        parse_layer_ids,
        parse_layer_weights,
    )


# ── Logging ───────────────────────────────────────────────────────────────────
class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return False

def setup_logging(output_dir: Path):
    log_file = open(output_dir / 'train.log', 'w', buffering=1)
    tee = _TeeIO(sys.__stdout__, log_file)
    sys.stdout = tee
    atexit.register(lambda: setattr(sys, 'stdout', sys.__stdout__))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_preinit(device: torch.device, retries: int = 10, sleep_s: float = 2.0):
    for i in range(retries):
        try:
            torch.zeros(1, device=device)
            return
        except RuntimeError:
            if i < retries - 1:
                time.sleep(sleep_s)
    raise RuntimeError(f'Cannot initialize {device}')


def make_loaders(batch_size: int, num_workers: int, smoke: bool, train_cache: str, val_cache: str):
    if smoke:
        ds = AugmentedCurriculumDataset(train_cache)
        sub = torch.utils.data.Subset(ds, range(min(40, len(ds))))
        train_loader = DataLoader(sub, batch_size=batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn_curriculum)
        val_loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate_fn_curriculum)
        return train_loader, val_loader

    train_ds = AugmentedCurriculumDataset(train_cache)
    val_ds = AugmentedCurriculumDataset(val_cache, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn_curriculum)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn_curriculum)
    return train_loader, val_loader


# ── 三段式 LR schedule ────────────────────────────────────────────────────────
def make_lr_lambda(warmup_epochs: int, constant_until: int, total_epochs: int, min_lr: float, max_lr: float):
    """
    Epoch 1 ~ warmup_epochs:    linear warmup
    Epoch warmup_epochs+1 ~ constant_until: constant max_lr
    Epoch constant_until+1 ~ total_epochs:  cosine decay to min_lr
    """
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        if ep < constant_until:
            return 1.0
        progress = (ep - constant_until) / max(1, total_epochs - constant_until)
        return max(min_lr / max_lr, 0.5 * (1 + math.cos(math.pi * progress)))
    return lr_lambda


# ── 動態 lambda_anchor ────────────────────────────────────────────────────────
def get_lambda_anchor(epoch: int) -> float:
    """
    Epoch 1-50:    0.5  (降噪優先)
    Epoch 51-150:  1.5  (逐步加強)
    Epoch 151+:    3.0  (強力鎖定)
    """
    if epoch <= 50:
        return 0.5
    elif epoch <= 150:
        return 1.5
    else:
        return 3.0


# ── Train epoch ───────────────────────────────────────────────────────────────
def train_epoch(
    model, anchor_encoder, student_hooks, anchor_hooks, layer_weights,
    loader, optimizer, scaler, device, cfg, mr_stft, mel_fn, epoch
) -> Dict:
    model.train()
    model.teacher.backbone.eval()
    model.teacher.head.eval()
    model.student.train()
    anchor_encoder.eval()

    accum = cfg['grad_accum']
    lambda_wav = cfg['lambda_wav']
    lambda_stft = cfg['lambda_stft']
    lambda_mel = cfg['lambda_mel']
    lambda_anchor = get_lambda_anchor(epoch)

    sums = {k: 0.0 for k in ['total_loss', 'wav_mse', 'stft_sc', 'stft_mag', 'mel_loss', 'anchor_loss']}
    n = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [exp_0305c]')
    for bi, batch in enumerate(pbar):
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)
        if clean.dim() == 2: clean = clean.unsqueeze(1)

        if bi % accum == 0:
            optimizer.zero_grad()

        student_hooks.clear()
        anchor_hooks.clear()
        with torch.no_grad():
            _ = anchor_encoder(noisy)

        with autocast(enabled=cfg['use_amp']):
            out = model.forward_wav(clean, noisy)
            recon = out['recon_wav']
            T = min(recon.shape[-1], clean.shape[-1])
            r, c = recon[..., :T], clean[..., :T]

            wav_mse = F.mse_loss(r, c)
            sc, mag = mr_stft(r, c)
            mel = mel_fn(r, c)
            anchor = compute_anchor_loss(student_hooks.cache, anchor_hooks.cache, layer_weights)

            total = (
                lambda_wav * wav_mse
                + lambda_stft * (sc + mag)
                + lambda_mel * mel
                + lambda_anchor * anchor
            ) / accum

        if scaler is not None:
            scaler.scale(total).backward()
            if (bi + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
        else:
            total.backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg['grad_clip'])
                optimizer.step()

        sums['total_loss'] += total.item() * accum
        sums['wav_mse'] += wav_mse.item()
        sums['stft_sc'] += sc.item()
        sums['stft_mag'] += mag.item()
        sums['mel_loss'] += mel.item()
        sums['anchor_loss'] += anchor.item()
        n += 1

        pbar.set_postfix(
            total=f'{total.item()*accum:.4f}',
            wav=f'{wav_mse.item():.5f}',
            anchor=f'{anchor.item():.5f}',
            lam=f'{lambda_anchor:.1f}',
        )

    if n > 0:
        for k in sums:
            sums[k] /= n
    sums['lambda_anchor'] = lambda_anchor
    return sums


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model, anchor_encoder, student_hooks, anchor_hooks, layer_weights,
    loader, device, cfg, mr_stft, mel_fn, max_batches, epoch
) -> Dict:
    model.eval()
    anchor_encoder.eval()
    lambda_anchor = get_lambda_anchor(epoch)
    lambda_wav = cfg['lambda_wav']
    lambda_stft = cfg['lambda_stft']
    lambda_mel = cfg['lambda_mel']

    vals = {k: [] for k in ['val_wav_mse', 'val_noisy_mse', 'val_stft_sc', 'val_mel', 'val_anchor', 'val_total_loss']}

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)
        if clean.dim() == 2: clean = clean.unsqueeze(1)

        student_hooks.clear()
        anchor_hooks.clear()
        _ = anchor_encoder(noisy)
        out = model.forward_wav(clean, noisy)
        recon = out['recon_wav']
        T = min(recon.shape[-1], clean.shape[-1], noisy.shape[-1])
        r, c, n_wav = recon[..., :T], clean[..., :T], noisy[..., :T]

        wav_mse = F.mse_loss(r, c).item()
        vals['val_wav_mse'].append(wav_mse)
        vals['val_noisy_mse'].append(F.mse_loss(n_wav, c).item())
        sc, mag = mr_stft(r, c)
        sc_v, mag_v = sc.item(), mag.item()
        vals['val_stft_sc'].append(sc_v)
        mel_v = mel_fn(r, c).item()
        vals['val_mel'].append(mel_v)
        anchor_v = compute_anchor_loss(student_hooks.cache, anchor_hooks.cache, layer_weights).item()
        vals['val_anchor'].append(anchor_v)
        vals['val_total_loss'].append(
            lambda_wav * wav_mse
            + lambda_stft * (sc_v + mag_v)
            + lambda_mel * mel_v
            + lambda_anchor * anchor_v
        )

    return {k: float(np.mean(v)) if v else float('nan') for k, v in vals.items()}


# ── Plot curves ───────────────────────────────────────────────────────────────
def plot_curves(history: dict, output_dir: Path, epoch: int):
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    ep = list(range(1, len(history.get('train_total_loss', [])) + 1))
    if not ep:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'exp_0305c [tail_lock, dynamic λ, 3-phase LR] — Epoch {epoch}', fontsize=13)

    def _plot(ax, keys, title, log=False):
        for key, label, color in keys:
            vals = history.get(key, [])
            if vals:
                ax.plot(ep[:len(vals)], vals, color=color, label=label, alpha=0.85)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(fontsize=8)
        if log: ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0],
          [('train_total_loss', 'Train total', 'steelblue'), ('val_total_loss', 'Val total', 'tomato')],
          'Total Loss', log=True)
    _plot(axes[0, 1],
          [('train_wav_mse', 'Train MSE', 'blue'), ('val_wav_mse', 'Val MSE', 'red'), ('val_noisy_mse', 'Noisy MSE', 'orange')],
          'Wav MSE')
    _plot(axes[0, 2],
          [('train_anchor', 'Train anchor', 'purple'), ('val_anchor', 'Val anchor', 'violet'), ('lambda_anchor', 'λ_anchor', 'gray')],
          'Anchor Loss & Lambda')
    _plot(axes[1, 0], [('lr', 'LR', 'gray')], 'Learning Rate', log=True)

    ax = axes[1, 1]
    val_mse = history.get('val_wav_mse', [])
    noisy_mse = history.get('val_noisy_mse', [])
    n_pts = min(len(val_mse), len(noisy_mse))
    if n_pts > 0:
        impr = [(noisy_mse[i] - val_mse[i]) / (noisy_mse[i] + 1e-9) for i in range(n_pts)]
        ax.plot(range(1, n_pts + 1), impr, 'g-', label='MSE improvement', alpha=0.85)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Val MSE Improvement Ratio'); ax.set_xlabel('Epoch')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.text(0.5, 0.5,
            'exp_0305c 改良:\n'
            '1) 3-phase LR:\n   warmup(10ep) → const(ep11-150) → cosine(ep151-300)\n\n'
            '2) Dynamic λ_anchor:\n   ep1-50: 0.5\n   ep51-150: 1.5\n   ep151+: 3.0\n\n'
            '3) tail_lock: L16, L17\n\n'
            'Baseline: exp_0305b expA best=0.02590\n'
            '          exp_0305b expB best=0.02574',
            transform=ax.transAxes, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')

    plt.tight_layout()
    fname = output_dir / f'exp0305c_epoch{epoch:03d}_{date_str}_curves.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Plot] 已儲存 loss 曲線圖: {fname.name}')


# ── Save audio samples ────────────────────────────────────────────────────────
@torch.no_grad()
def save_audio_samples(model, val_loader, device, output_dir, epoch, n_samples=4):
    audio_dir = output_dir / 'audio_samples' / f'epoch{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in val_loader:
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)
        if clean.dim() == 2: clean = clean.unsqueeze(1)
        out = model.forward_wav(clean, noisy)
        recon = out['recon_wav']
        for i in range(min(noisy.shape[0], n_samples - saved)):
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_noisy.wav'), noisy[i].cpu(), SAMPLE_RATE)
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_clean.wav'), clean[i].cpu(), SAMPLE_RATE)
            r = recon[i].cpu()
            if r.dim() == 1: r = r.unsqueeze(0)
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_recon.wav'), r, SAMPLE_RATE)
        saved += min(noisy.shape[0], n_samples - saved)
        if saved >= n_samples:
            break
    print(f'  [Audio] 已儲存 {saved} 筆音檔至 {audio_dir}')


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='exp_0305c: dynamic lambda + 3-phase LR')
    p.add_argument('--mode', default='smoke', choices=['smoke', 'epoch'])
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--grad_accum', type=int, default=2)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--constant_until', type=int, default=150,
                   help='constant LR 維持到第幾 epoch，之後才開始 cosine decay')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--use_amp', dest='use_amp', action='store_true')
    p.add_argument('--no_amp', dest='use_amp', action='store_false')
    p.set_defaults(use_amp=True)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--train_cache', default=str(TRAIN_CACHE))
    p.add_argument('--val_cache', default=str(VAL_CACHE))
    p.add_argument('--lambda_wav', type=float, default=1.0)
    p.add_argument('--lambda_stft', type=float, default=1.0)
    p.add_argument('--lambda_mel', type=float, default=45.0)
    p.add_argument('--lora_rank', type=int, default=64)
    p.add_argument('--lora_alpha', type=int, default=128)
    p.add_argument('--preset', default='tail_lock',
                   choices=['tail_lock', 'front_lock', 'front_tail_lock', 'stable_lock', 'custom'])
    p.add_argument('--anchor_layers', default='16,17')
    p.add_argument('--anchor_weights', default='')
    p.add_argument('--eval_max_batches', type=int, default=30)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    smoke = args.mode == 'smoke'
    if smoke:
        args.epochs = min(args.epochs, 5)
        args.eval_max_batches = 5

    layer_ids = parse_layer_ids(args.anchor_layers) if args.preset == 'custom' else preset_to_layers(args.preset)
    layer_weights = parse_layer_weights(args.anchor_weights, layer_ids)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != 'cuda' and args.use_amp:
        print('[Info] AMP disabled (not CUDA)')
        args.use_amp = False
    cuda_preinit(device)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(__file__).parent / 'runs' / f'exp0305c_{args.preset}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    cfg = vars(args).copy()
    cfg.update({'timestamp': ts, 'anchor_layer_ids': layer_ids, 'anchor_layer_weights': layer_weights,
                'experiment': 'exp_0305c', 'lr_schedule': '3-phase: warmup+constant+cosine',
                'lambda_schedule': 'dynamic: ep1-50=0.5, ep51-150=1.5, ep151+=3.0'})
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print('=' * 70)
    print('exp_0305c: 官方 WavTokenizer 起點 + dynamic λ + 3-phase LR')
    print(f'mode={args.mode}, epochs={args.epochs}, device={device}')
    print(f'preset={args.preset}, anchor_layers={layer_ids}')
    print(f'LR: warmup {args.warmup_epochs}ep → const until ep{args.constant_until} → cosine → {args.min_lr}')
    print(f'λ_anchor: ep1-50=0.5, ep51-150=1.5, ep151+=3.0')
    print(f'output={out_dir}')
    print('=' * 70)

    train_loader, val_loader = make_loaders(
        batch_size=(4 if smoke else args.batch_size),
        num_workers=(0 if smoke else args.num_workers),
        smoke=smoke,
        train_cache=args.train_cache,
        val_cache=args.val_cache,
    )

    # Model：官方 WavTokenizer 起點
    model = TeacherStudentNoVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6],
        device=str(device),
    ).to(device)
    print('Student encoder: 官方 WavTokenizer 預訓練權重（NO 0224a/0217 先驗）')

    # Anchor encoder = 官方 WavTokenizer teacher encoder（frozen）
    anchor_encoder = model.teacher.feature_extractor.encodec.encoder.eval()
    for p in anchor_encoder.parameters():
        p.requires_grad_(False)

    student_enc = model.student.feature_extractor.encodec.encoder
    anchor_enc = model.teacher.feature_extractor.encodec.encoder
    student_modules = build_conv18_modules(student_enc)
    anchor_modules = build_conv18_modules(anchor_enc)
    student_hooks = LayerHookBank(student_modules, layer_ids)
    anchor_hooks = LayerHookBank(anchor_modules, layer_ids)
    print(f'[Anchor] 錨定層 = {layer_ids}（官方 WavTokenizer encoder）')

    mr_stft = MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[512, 256, 128],
        win_sizes=[2048, 1024, 512],
    ).to(device)
    mel_fn = MelReconstructionLoss(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    # 三段式 LR schedule
    lr_lambda = make_lr_lambda(
        warmup_epochs=args.warmup_epochs,
        constant_until=args.constant_until,
        total_epochs=args.epochs,
        min_lr=args.min_lr,
        max_lr=args.learning_rate,
    )
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler(enabled=args.use_amp) if args.use_amp else None

    history = {
        'train_total_loss': [], 'train_wav_mse': [], 'train_anchor': [], 'lambda_anchor': [],
        'val_total_loss': [], 'val_wav_mse': [], 'val_noisy_mse': [], 'val_anchor': [], 'lr': [],
    }
    best_val = float('inf')

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(
            model, anchor_encoder, student_hooks, anchor_hooks, layer_weights,
            train_loader, opt, scaler, device, cfg, mr_stft, mel_fn, ep
        )
        va = evaluate(
            model, anchor_encoder, student_hooks, anchor_hooks, layer_weights,
            val_loader, device, cfg, mr_stft, mel_fn, args.eval_max_batches, ep
        )
        sch.step()
        lr = opt.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_total_loss'].append(tr['total_loss'])
        history['train_wav_mse'].append(tr['wav_mse'])
        history['train_anchor'].append(tr['anchor_loss'])
        history['lambda_anchor'].append(tr['lambda_anchor'])
        history['val_total_loss'].append(va['val_total_loss'])
        history['val_wav_mse'].append(va['val_wav_mse'])
        history['val_noisy_mse'].append(va['val_noisy_mse'])
        history['val_anchor'].append(va['val_anchor'])
        history['lr'].append(lr)

        improve = (va['val_noisy_mse'] - va['val_wav_mse']) / (va['val_noisy_mse'] + 1e-9) * 100.0
        print(
            f"Epoch {ep}/{args.epochs} ({elapsed:.1f}s) "
            f"λ={tr['lambda_anchor']:.1f} "
            f"train_total={tr['total_loss']:.4f} train_anchor={tr['anchor_loss']:.5f} "
            f"val_mse={va['val_wav_mse']:.5f} noisy={va['val_noisy_mse']:.5f} "
            f"improve=+{improve:.2f}% val_anchor={va['val_anchor']:.5f} lr={lr:.3e}"
        )

        if va['val_wav_mse'] < best_val:
            best_val = va['val_wav_mse']
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': sch.state_dict(),
                'metrics': va, 'config': cfg,
            }, out_dir / 'best_model.pt')
            print(f'  New best val_wav_mse={best_val:.6f}')

        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # 每 25 epoch：loss 圖 + 音檔 + checkpoint
        if ep % 25 == 0 or (smoke and ep == args.epochs):
            plot_curves(history, out_dir, ep)
            save_audio_samples(model, val_loader, device, out_dir, ep)
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': sch.state_dict(),
                'metrics': va, 'config': cfg,
            }, out_dir / f'checkpoint_epoch{ep:03d}.pt')

    student_hooks.close()
    anchor_hooks.close()
    plot_curves(history, out_dir, args.epochs)
    print(f'Training complete. Best val_wav_mse={best_val:.6f}')
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    main()

"""
exp_0223: Decoder LoRA Fine-tune 訓練腳本

策略：
    - Encoder + VQ: 凍結（繼承 exp_0217 best_model.pt）
    - Decoder backbone ConvNeXt pwconv1/pwconv2: 加 LoRA (rank=32)，可訓練
    - Loss: MSE(recon_wav, clean_wav)  — 純 wav-domain 端對端監督

執行：
    # Smoke test (5 epochs, 20 samples)
    /home/sbplab/miniconda3/envs/test/bin/python families/eval/decoder_lora_eval/train_decoder_lora.py \\
        --mode smoke --epochs 5

    # Long-run (150 epochs)
    /home/sbplab/miniconda3/envs/test/bin/python families/eval/decoder_lora_eval/train_decoder_lora.py \\
        --mode epoch --epochs 150 --device cuda:1
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
import sys
import gc
import math
import time
import atexit
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from families.eval.decoder_lora_eval.models_decoder_lora import TeacherStudentDecoderLoRA
from families.deps.encoder_aug.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum


# ============================================================
# Constants
# ============================================================

EXP0217_BEST_CKPT = (
    Path(__file__).parent.parent /
    'families/deps/t453_weighted_baseline/runs/t453_weighted_epoch_20260217_104843/best_model.pt'
)

# ============================================================
# Utilities
# ============================================================

class _TeeIO:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def setup_logging(output_dir: Path) -> Path:
    log_path = output_dir / "train.log"
    try:
        log_f = open(log_path, "a", buffering=1, encoding="utf-8", errors="ignore")
    except Exception:
        return None
    atexit.register(lambda: log_f.close())
    sys.stdout = _TeeIO(sys.stdout, log_f)
    sys.stderr = _TeeIO(sys.stderr, log_f)
    return log_path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Fixed seed={seed}")


def cuda_preinit(device: torch.device, retries: int = 10, sleep_s: float = 2.0):
    if device.type != 'cuda':
        return
    for attempt in range(retries):
        try:
            torch.zeros(1, device=device)
            print(f"CUDA pre-init OK (attempt {attempt + 1})")
            return
        except RuntimeError as e:
            print(f"CUDA pre-init attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(sleep_s)
    raise RuntimeError(f"CUDA pre-init failed after {retries} attempts")


def make_val_loader(val_cache_path, batch_size=4, num_workers=2):
    ds = AugmentedCurriculumDataset(
        val_cache_path,
        augment=False,
        filter_clean_to_clean=True,
        compute_snr=False,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_curriculum,
    )


def make_train_loader(train_cache_path, batch_size=8, num_workers=2,
                      snr_remix_prob=0.5, snr_remix_range=(-5.0, 25.0),
                      random_gain_prob=0.3, random_gain_db=3.0,
                      random_crop_prob=0.3, random_crop_min_ratio=0.7,
                      time_stretch_prob=0.2, time_stretch_range=(0.95, 1.05)):
    ds = AugmentedCurriculumDataset(
        train_cache_path,
        augment=True,
        filter_clean_to_clean=True,
        compute_snr=False,
        snr_remix_prob=snr_remix_prob,
        snr_remix_range=snr_remix_range,
        random_gain_prob=random_gain_prob,
        random_gain_db=random_gain_db,
        random_crop_prob=random_crop_prob,
        random_crop_min_ratio=random_crop_min_ratio,
        time_stretch_prob=time_stretch_prob,
        time_stretch_range=time_stretch_range,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_curriculum,
    )


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, dataloader, optimizer, device, epoch, config,
                scaler=None) -> Dict:
    """Decoder LoRA 訓練 epoch

    loss = MSE(recon_wav, clean_wav)
    encoder + VQ 全部 no_grad，只有 decoder LoRA 接收 gradient。
    """
    model.train()
    # 確保 encoder / VQ 不收到梯度（雙重保護）
    model.teacher.backbone.train()  # LoRA dropout active
    model.teacher.head.eval()       # head 沒有 LoRA，不需要 train mode

    metrics = {
        'total_loss': 0.0,
        'nan_batches': 0,
    }
    n_batches = 0
    nan_count = 0
    max_nan_per_epoch = 10

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [decoder LoRA]")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        if batch_idx % config['grad_accum'] == 0:
            optimizer.zero_grad()

        with autocast(enabled=config['use_amp']):
            out = model.forward_wav(clean_audio, noisy_audio)
            recon_wav = out['recon_wav']

            # align length
            T_clean = clean_audio.shape[-1]
            T_recon = recon_wav.shape[-1]
            T = min(T_clean, T_recon)

            loss = F.mse_loss(recon_wav[..., :T], clean_audio[..., :T])
            loss = loss / config['grad_accum']

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            metrics['nan_batches'] = nan_count
            print(f"  NaN/Inf at batch {batch_idx}, skipping (count: {nan_count})")
            optimizer.zero_grad()
            if nan_count >= max_nan_per_epoch:
                print(f"  Too many NaN batches ({nan_count}), aborting epoch!")
                break
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip'],
                )
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip'],
                )
                optimizer.step()

        loss_val = loss.item() * config['grad_accum']
        metrics['total_loss'] += loss_val
        n_batches += 1

        pbar.set_postfix({'wav_mse': f"{loss_val:.5f}"})

    if n_batches > 0:
        metrics['total_loss'] /= n_batches

    return metrics


@torch.no_grad()
def evaluate_decoder(model, dataloader, device, config, max_batches=30) -> Dict:
    """Decoder LoRA 評估

    計算 val 集上的 wav MSE(recon, clean) 和 wav MSE(noisy, clean)，
    並嘗試計算 PESQ/STOI（如果 pesq/pystoi 可用）。
    """
    model.eval()

    wav_mse_list = []
    noisy_mse_list = []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        out = model.forward_wav(clean_audio, noisy_audio)
        recon_wav = out['recon_wav']

        T = min(clean_audio.shape[-1], recon_wav.shape[-1], noisy_audio.shape[-1])
        clean_t = clean_audio[..., :T]
        recon_t = recon_wav[..., :T]
        noisy_t = noisy_audio[..., :T]

        wav_mse_list.append(F.mse_loss(recon_t, clean_t).item())
        noisy_mse_list.append(F.mse_loss(noisy_t, clean_t).item())

    model.train()

    val_wav_mse = float(np.mean(wav_mse_list)) if wav_mse_list else float('nan')
    val_noisy_mse = float(np.mean(noisy_mse_list)) if noisy_mse_list else float('nan')

    return {
        'val_wav_mse': val_wav_mse,
        'val_noisy_mse': val_noisy_mse,
    }


def plot_training_curves(history, output_dir, epoch):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'exp_0223: Decoder LoRA Fine-tune (Epoch {epoch})', fontsize=14)

    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train wav MSE', alpha=0.8)
    if history.get('val_wav_mse'):
        ax.plot(epochs, history['val_wav_mse'], 'r-', label='Val wav MSE', alpha=0.8)
    ax.set_title('Wav-domain MSE (recon vs clean)')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    if history.get('val_noisy_mse'):
        ax.plot(epochs, history['val_noisy_mse'], 'gray', ls='--',
                label='Noisy vs Clean MSE (baseline)', alpha=0.8)
    if history.get('val_wav_mse'):
        ax.plot(epochs, history['val_wav_mse'], 'r-', label='Recon vs Clean MSE', alpha=0.8)
    ax.set_title('Recon MSE vs Noisy MSE (val)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'green', linewidth=2)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    ax = axes[1, 1]
    if history.get('val_wav_mse') and history.get('val_noisy_mse'):
        improvement = [
            (nm - rm) / nm * 100
            for nm, rm in zip(history['val_noisy_mse'], history['val_wav_mse'])
        ]
        ax.plot(epochs[:len(improvement)], improvement, 'purple', linewidth=2)
        ax.axhline(y=0, color='gray', ls='--')
        ax.set_title('MSE Improvement over Noisy (%)')
        ax.set_xlabel('Epoch')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch{epoch:03d}.png', dpi=150)
    plt.close()
    print(f"  Loss plot saved: training_curves_epoch{epoch:03d}.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='exp_0223: Decoder LoRA Fine-tune'
    )

    parser.add_argument('--mode', type=str, default='smoke',
                        choices=['smoke', 'epoch'],
                        help='smoke=快速驗證(5 epochs), epoch=正式訓練')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--output_dir', type=str, default=None)

    # Training basics
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:1')

    # Encoder LoRA（用於 checkpoint 載入，不訓練）
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    # Decoder LoRA
    parser.add_argument('--decoder_lora_rank', type=int, default=32)
    parser.add_argument('--decoder_lora_alpha', type=int, default=64)
    parser.add_argument('--decoder_lora_dropout', type=float, default=0.1)

    # Encoder checkpoint
    parser.add_argument(
        '--encoder_ckpt', type=str,
        default=str(EXP0217_BEST_CKPT),
        help='exp_0217 best_model.pt 路徑',
    )

    # Data augmentation
    parser.add_argument('--snr_remix_prob', type=float, default=0.5)
    parser.add_argument('--snr_remix_min', type=float, default=-5.0)
    parser.add_argument('--snr_remix_max', type=float, default=25.0)
    parser.add_argument('--random_gain_prob', type=float, default=0.3)
    parser.add_argument('--random_gain_db', type=float, default=3.0)
    parser.add_argument('--random_crop_prob', type=float, default=0.3)
    parser.add_argument('--random_crop_min_ratio', type=float, default=0.7)
    parser.add_argument('--time_stretch_prob', type=float, default=0.2)
    parser.add_argument('--time_stretch_min', type=float, default=0.95)
    parser.add_argument('--time_stretch_max', type=float, default=1.05)

    # Evaluation & Saving
    parser.add_argument('--save_checkpoint_every', type=int, default=10)
    parser.add_argument('--save_audio_interval', type=int, default=25)
    parser.add_argument('--eval_max_batches', type=int, default=30)

    args = parser.parse_args()

    # Smoke test: override to tiny run
    if args.mode == 'smoke':
        args.epochs = max(args.epochs, 5)
        args.eval_max_batches = 5

    # ===== Setup =====
    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(f'families/eval/decoder_lora_eval/runs/decoder_lora_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0223_decoder_lora'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0223: Decoder LoRA Fine-tune")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed} (fixed)")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LR: {args.learning_rate} → {args.min_lr}")
    print(f"Decoder LoRA: rank={args.decoder_lora_rank}, alpha={args.decoder_lora_alpha}")
    print(f"Encoder LoRA: rank={args.lora_rank} (frozen)")
    print(f"Encoder ckpt: {args.encoder_ckpt}")
    print(f"Loss: wav-domain MSE(recon, clean)")
    print(f"Output: {exp_dir}")
    if log_path:
        print(f"Log: {log_path}")
    print("=" * 70)

    # ===== CUDA =====
    device = torch.device(args.device)
    cuda_preinit(device)

    # ===== Data =====
    print("\nLoading data...")
    if args.mode == 'smoke':
        # Smoke test: 使用 val set 的小子集
        from torch.utils.data import Subset
        full_ds = AugmentedCurriculumDataset(
            VAL_CACHE, augment=False,
            filter_clean_to_clean=True, compute_snr=False,
        )
        smoke_indices = list(range(min(20, len(full_ds))))
        smoke_ds = Subset(full_ds, smoke_indices)
        train_loader = DataLoader(
            smoke_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=0,
            collate_fn=collate_fn_curriculum,
        )
        val_loader = DataLoader(
            smoke_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
            collate_fn=collate_fn_curriculum,
        )
        print(f"Smoke test: {len(smoke_ds)} samples")
    else:
        train_loader = make_train_loader(
            TRAIN_CACHE,
            batch_size=args.batch_size,
            num_workers=2,
            snr_remix_prob=args.snr_remix_prob,
            snr_remix_range=(args.snr_remix_min, args.snr_remix_max),
            random_gain_prob=args.random_gain_prob,
            random_gain_db=args.random_gain_db,
            random_crop_prob=args.random_crop_prob,
            random_crop_min_ratio=args.random_crop_min_ratio,
            time_stretch_prob=args.time_stretch_prob,
            time_stretch_range=(args.time_stretch_min, args.time_stretch_max),
        )
        val_loader = make_val_loader(VAL_CACHE, batch_size=4, num_workers=2)
        print(f"Train: {len(train_loader.dataset)} samples, "
              f"Val: {len(val_loader.dataset)} samples")

    # ===== Model =====
    print("\nBuilding TeacherStudentDecoderLoRA...")
    model = TeacherStudentDecoderLoRA(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6],
        device=device,
        decoder_lora_rank=args.decoder_lora_rank,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_lora_dropout=args.decoder_lora_dropout,
    ).to(device)

    # ===== Load Encoder Checkpoint =====
    encoder_ckpt_path = Path(args.encoder_ckpt)
    if encoder_ckpt_path.exists():
        print(f"\nLoading encoder checkpoint: {encoder_ckpt_path}")
        model.load_encoder_vq_checkpoint(str(encoder_ckpt_path))
    else:
        print(f"\n[WARNING] Encoder checkpoint not found: {encoder_ckpt_path}")
        print("  Starting with fresh encoder LoRA (not recommended for production)")

    # ===== 驗證 trainable params =====
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"\nTrainable params: {trainable_count:,} / {total_params:,} "
          f"({100*trainable_count/total_params:.3f}%)")

    # 確認只有 decoder LoRA 可訓練
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_decoder_lora = sum(1 for n in trainable_names if 'lora_' in n)
    print(f"Trainable LoRA params: {n_decoder_lora} tensors")
    assert all('lora_' in n or 'modules_to_save' in n for n in trainable_names), \
        f"Unexpected trainable params: {[n for n in trainable_names if 'lora_' not in n]}"
    print("  OK: Only decoder LoRA params are trainable")

    # ===== Optimizer + Scheduler =====
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.learning_rate,
                   0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if args.use_amp else None

    # ===== Training History =====
    history = {
        'train_loss': [],
        'val_wav_mse': [],
        'val_noisy_mse': [],
        'lr': [],
    }
    best_val_mse = float('inf')

    # ===== Training Loop =====
    print(f"\nStarting training ({args.epochs} epochs)...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            config=config, scaler=scaler,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_metrics = evaluate_decoder(
            model, val_loader, device, config,
            max_batches=args.eval_max_batches,
        )

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_metrics['total_loss'])
        history['val_wav_mse'].append(val_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(val_metrics['val_noisy_mse'])
        history['lr'].append(current_lr)

        # Improvement over noisy
        noisy_mse = val_metrics['val_noisy_mse']
        recon_mse = val_metrics['val_wav_mse']
        improvement_pct = (noisy_mse - recon_mse) / noisy_mse * 100 if noisy_mse > 0 else 0.0

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: wav_mse={train_metrics['total_loss']:.5f}")
        print(f"  Val:   recon_mse={recon_mse:.5f}  noisy_mse={noisy_mse:.5f}  "
              f"improvement={improvement_pct:+.1f}%")
        print(f"  LR={current_lr:.2e}")

        # Save best model
        if recon_mse < best_val_mse:
            best_val_mse = recon_mse
            torch.save({
                'epoch': epoch,
                'decoder_lora_state': model.get_decoder_lora_state_dict(),
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  New best val MSE: {best_val_mse:.5f} → saved best_model.pt")

        # Periodic checkpoint
        if epoch % args.save_checkpoint_every == 0:
            ckpt_dir = exp_dir / 'checkpoints'
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'decoder_lora_state': model.get_decoder_lora_state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, ckpt_dir / f'checkpoint_epoch{epoch:03d}.pt')

        # Save history
        with open(exp_dir / 'metrics_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Periodic plot + audio
        if epoch % args.save_audio_interval == 0 or epoch == 1 or epoch == args.epochs:
            try:
                plot_training_curves(history, exp_dir, epoch)
            except Exception as e:
                print(f"  Warning: Plot failed: {e}")

            try:
                _save_audio_samples(model, val_loader, device, exp_dir, epoch, split='val')
            except Exception as e:
                print(f"  Warning: Val audio save failed: {e}")

            try:
                _save_audio_samples(model, train_loader, device, exp_dir, epoch, split='train')
            except Exception as e:
                print(f"  Warning: Train audio save failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()

    # ===== Final save =====
    torch.save({
        'epoch': args.epochs,
        'decoder_lora_state': model.get_decoder_lora_state_dict(),
        'model_state_dict': model.state_dict(),
        'config': config,
    }, exp_dir / 'final_model.pt')

    try:
        plot_training_curves(history, exp_dir, args.epochs)
    except Exception as e:
        print(f"  Warning: Final plot failed: {e}")

    summary = {
        'experiment': 'exp_0223_decoder_lora',
        'mode': args.mode,
        'total_epochs': args.epochs,
        'seed': args.seed,
        'best_val_mse': best_val_mse,
        'config': config,
        'final_metrics': {
            'train_loss': history['train_loss'][-1],
            'val_wav_mse': history['val_wav_mse'][-1],
            'val_noisy_mse': history['val_noisy_mse'][-1],
        },
        'baseline_reference': {
            'noisy_teacher_vq_pesq': 1.203,
            'clean_teacher_vq_pesq': 1.790,
            'exp_0217_pesq': 1.147,
            'exp_0217_stoi': 0.511,
        },
    }
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"exp_0223 training done!")
    print(f"  Best val MSE: {best_val_mse:.5f}")
    print(f"  Results: {exp_dir}")
    print(f"{'='*70}")


def _save_audio_samples(model, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    """儲存 noisy / clean / recon 音檔（使用 scipy 避免 torchaudio 問題）"""
    try:
        import scipy.io.wavfile as wav
        import numpy as np
    except ImportError:
        return

    model.eval()
    audio_dir = output_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    SR = 24000
    data_iter = iter(loader)

    for i in range(min(num_samples, len(loader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        with torch.no_grad():
            noisy = batch['noisy_audio'][:1].to(device)
            clean = batch['clean_audio'][:1].to(device)
            if noisy.dim() == 2: noisy = noisy.unsqueeze(1)
            if clean.dim() == 2: clean = clean.unsqueeze(1)

            out = model.forward_wav(clean, noisy)
            recon = out['recon_wav']

        def to_np(t):
            x = t.squeeze().cpu().numpy()
            x = np.clip(x, -1.0, 1.0)
            return (x * 32767).astype(np.int16)

        wav.write(str(audio_dir / f'sample_{i+1}_noisy.wav'), SR, to_np(noisy))
        wav.write(str(audio_dir / f'sample_{i+1}_clean.wav'), SR, to_np(clean))
        wav.write(str(audio_dir / f'sample_{i+1}_recon.wav'), SR, to_np(recon))

    model.train()
    print(f"  Audio saved ({split}) → {audio_dir}")


if __name__ == '__main__':
    main()

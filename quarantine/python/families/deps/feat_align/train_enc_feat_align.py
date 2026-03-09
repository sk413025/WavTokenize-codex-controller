"""
exp_0226a: No-VQ Encoder LoRA + Feature Alignment Loss

架構（與 exp_0225a 完全相同）：
    Encoder LoRA (Trainable, 初始化自 exp_0225a best_model_val_total.pt):
        Noisy Audio → LoRA Encoder → student_encoder_out [B, 512, T]

    Decoder (Frozen, WavTokenizer pretrained):
        student_encoder_out → backbone → head → recon_wav

Loss（與 exp_0225a 的差異）：
    λ_wav  * MSE(recon_wav, clean_wav)
    + λ_stft * MR-STFT(recon_wav, clean_wav)
    + λ_mel  * Mel(recon_wav, clean_wav)
    + λ_feat * MSE(student_encoder_out, teacher_encoder_out)   ← 新增

設計動機：
    exp_0225a 的 val_mse 在 ep43 最低後反彈 → encoder feature 偏離 pretrained 分佈。
    加入 feature alignment 讓 encoder 在改善重建品質的同時，
    輸出保持接近 teacher encoder（clean），使 frozen decoder 不會迷失。
    理論天花板：noisy_through_teacher_no_vq PESQ ≈ 1.707。

執行：
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python families/deps/feat_align/train_enc_feat_align.py --mode smoke

    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python families/deps/feat_align/train_enc_feat_align.py \\
        --mode epoch --epochs 300 --device cuda:1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
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
import scipy.io.wavfile as wavfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from families.deps.no_vq_core.models_no_vq import TeacherStudentNoVQ
from families.deps.encoder_aug.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum


# ============================================================
# Constants
# ============================================================

SAMPLE_RATE = 24000

EXP0225A_BEST_CKPT = (
    'families/deps/no_vq_scratch/runs/no_vq_scratch_epoch_20260224_032104/best_model_val_total.pt'
)


# ============================================================
# Multi-Resolution STFT Loss
# ============================================================

class STFTLoss(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        )
        return torch.abs(spec)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag_hat = self._stft(y_hat)
        mag = self._stft(y)
        sc_loss = torch.norm(mag - mag_hat, p='fro') / (torch.norm(mag, p='fro') + 1e-7)
        log_mag_loss = F.l1_loss(
            torch.log(mag.clamp(min=1e-7)),
            torch.log(mag_hat.clamp(min=1e-7)),
        )
        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512],
        hop_sizes: List[int] = [512, 256, 128],
        win_sizes: List[int] = [2048, 1024, 512],
    ):
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft, hop, win)
            for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        sc_loss, mag_loss = 0.0, 0.0
        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(y_hat, y)
            sc_loss += sc
            mag_loss += mag
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss


class MelReconstructionLoss(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, center=True, power=1,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        mel_hat = torch.log(self.mel_spec(y_hat).clamp(min=1e-7))
        mel = torch.log(self.mel_spec(y).clamp(min=1e-7))
        return F.l1_loss(mel, mel_hat)


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
        val_cache_path, augment=False,
        filter_clean_to_clean=True, compute_snr=False,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_curriculum,
    )


def make_train_loader(train_cache_path, batch_size=8, num_workers=2,
                      snr_remix_prob=0.5, snr_remix_range=(-5.0, 25.0),
                      random_gain_prob=0.3, random_gain_db=3.0,
                      random_crop_prob=0.3, random_crop_min_ratio=0.7,
                      time_stretch_prob=0.2, time_stretch_range=(0.95, 1.05)):
    ds = AugmentedCurriculumDataset(
        train_cache_path, augment=True,
        filter_clean_to_clean=True, compute_snr=False,
        snr_remix_prob=snr_remix_prob, snr_remix_range=snr_remix_range,
        random_gain_prob=random_gain_prob, random_gain_db=random_gain_db,
        random_crop_prob=random_crop_prob, random_crop_min_ratio=random_crop_min_ratio,
        time_stretch_prob=time_stretch_prob, time_stretch_range=time_stretch_range,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_curriculum,
    )


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, dataloader, optimizer, device, epoch, config,
                mr_stft_loss_fn, mel_loss_fn, scaler=None) -> Dict:
    model.train()
    model.teacher.backbone.eval()
    model.teacher.head.eval()
    model.student.train()

    metrics = {
        'total_loss': 0.0,
        'wav_mse': 0.0,
        'stft_sc_loss': 0.0,
        'stft_mag_loss': 0.0,
        'mel_loss': 0.0,
        'feat_align_loss': 0.0,
        'nan_batches': 0,
    }
    n_batches = 0
    nan_count = 0
    max_nan_per_epoch = 10

    lambda_wav = config['lambda_wav']
    lambda_stft = config['lambda_stft']
    lambda_mel = config['lambda_mel']
    lambda_feat = config['lambda_feat']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [enc+feat-align]")

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
            student_feat = out['student_encoder_out']      # [B, 512, T]
            teacher_feat = out['teacher_encoder_out']      # [B, 512, T] — no_grad in model

            T = min(clean_audio.shape[-1], recon_wav.shape[-1])
            recon_t = recon_wav[..., :T]
            clean_t = clean_audio[..., :T]

            # waveform-domain losses（同 0225a）
            wav_mse = F.mse_loss(recon_t, clean_t)
            sc_loss, mag_loss = mr_stft_loss_fn(recon_t, clean_t)
            stft_loss = sc_loss + mag_loss
            mel_loss = mel_loss_fn(recon_t, clean_t)

            # feature alignment loss（新增）
            # teacher_feat 已在 model.forward_wav 內以 no_grad 計算，這裡直接 detach 確保
            Tf = min(student_feat.shape[-1], teacher_feat.shape[-1])
            feat_align = F.mse_loss(
                student_feat[..., :Tf],
                teacher_feat[..., :Tf].detach(),
            )

            loss = (
                lambda_wav  * wav_mse
                + lambda_stft * stft_loss
                + lambda_mel  * mel_loss
                + lambda_feat * feat_align
            )
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
        metrics['wav_mse'] += wav_mse.item()
        metrics['stft_sc_loss'] += sc_loss.item()
        metrics['stft_mag_loss'] += mag_loss.item()
        metrics['mel_loss'] += mel_loss.item()
        metrics['feat_align_loss'] += feat_align.item()
        n_batches += 1

        pbar.set_postfix({
            'total': f"{loss_val:.4f}",
            'wav': f"{wav_mse.item():.5f}",
            'feat': f"{feat_align.item():.5f}",
            'mel': f"{mel_loss.item():.3f}",
        })

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
                  'mel_loss', 'feat_align_loss']:
            metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, config,
             mr_stft_loss_fn, mel_loss_fn, max_batches=30) -> Dict:
    model.eval()

    wav_mse_list, noisy_mse_list = [], []
    stft_sc_list, stft_mag_list, mel_list = [], [], []
    noisy_stft_sc_list, noisy_mel_list = [], []
    feat_align_list = []

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
        student_feat = out['student_encoder_out']
        teacher_feat = out['teacher_encoder_out']

        T = min(clean_audio.shape[-1], recon_wav.shape[-1], noisy_audio.shape[-1])
        clean_t = clean_audio[..., :T]
        recon_t = recon_wav[..., :T]
        noisy_t = noisy_audio[..., :T]

        wav_mse_list.append(F.mse_loss(recon_t, clean_t).item())
        sc, mag = mr_stft_loss_fn(recon_t, clean_t)
        stft_sc_list.append(sc.item())
        stft_mag_list.append(mag.item())
        mel_list.append(mel_loss_fn(recon_t, clean_t).item())

        noisy_mse_list.append(F.mse_loss(noisy_t, clean_t).item())
        sc_n, _ = mr_stft_loss_fn(noisy_t, clean_t)
        noisy_stft_sc_list.append(sc_n.item())
        noisy_mel_list.append(mel_loss_fn(noisy_t, clean_t).item())

        Tf = min(student_feat.shape[-1], teacher_feat.shape[-1])
        feat_align_list.append(
            F.mse_loss(student_feat[..., :Tf], teacher_feat[..., :Tf]).item()
        )

    model.train()

    return {
        'val_wav_mse':       float(np.mean(wav_mse_list))       if wav_mse_list else float('nan'),
        'val_noisy_mse':     float(np.mean(noisy_mse_list))     if noisy_mse_list else float('nan'),
        'val_stft_sc':       float(np.mean(stft_sc_list))       if stft_sc_list else float('nan'),
        'val_stft_mag':      float(np.mean(stft_mag_list))      if stft_mag_list else float('nan'),
        'val_mel_loss':      float(np.mean(mel_list))           if mel_list else float('nan'),
        'val_noisy_stft_sc': float(np.mean(noisy_stft_sc_list)) if noisy_stft_sc_list else float('nan'),
        'val_noisy_mel':     float(np.mean(noisy_mel_list))     if noisy_mel_list else float('nan'),
        'val_feat_align':    float(np.mean(feat_align_list))    if feat_align_list else float('nan'),
    }


def _save_audio_samples(model, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    audio_dir = output_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    with torch.no_grad():
        for batch in loader:
            if saved >= num_samples:
                break
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)

            if clean_audio.dim() == 2:
                clean_audio = clean_audio.unsqueeze(1)
            if noisy_audio.dim() == 2:
                noisy_audio = noisy_audio.unsqueeze(1)

            out = model.forward_wav(clean_audio, noisy_audio)
            recon_wav = out['recon_wav']

            B = min(noisy_audio.shape[0], num_samples - saved)
            for b in range(B):
                def _save(tensor, name):
                    wav = tensor[b].squeeze().cpu().float().numpy()
                    wav = np.clip(wav, -1.0, 1.0)
                    wav_int16 = (wav * 32767).astype(np.int16)
                    wavfile.write(str(audio_dir / name), SAMPLE_RATE, wav_int16)

                _save(noisy_audio, f'sample{saved+b:02d}_noisy.wav')
                T = min(clean_audio.shape[-1], recon_wav.shape[-1])
                _save(recon_wav[..., :T], f'sample{saved+b:02d}_recon.wav')
                _save(clean_audio[..., :T], f'sample{saved+b:02d}_clean.wav')

            saved += B

    model.train()
    print(f"  Audio saved ({split}) → {audio_dir}")


def plot_training_curves(history, output_dir, epoch):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'exp_0226a: Encoder LoRA + Feature Alignment (Epoch {epoch})', fontsize=14)

    epochs = range(1, len(history['train_total_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train total', alpha=0.8)
    ax.set_title('Total Loss (train)')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    if history.get('val_noisy_mse'):
        ax.plot(epochs, history['val_noisy_mse'], 'gray', ls='--',
                label='Noisy baseline', alpha=0.8)
    if history.get('val_wav_mse'):
        ax.plot(epochs, history['val_wav_mse'], 'r-', label='Recon vs Clean', alpha=0.8)
    ax.set_title('Wav MSE (val)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    if history.get('train_stft_sc'):
        ax.plot(epochs, history['train_stft_sc'], 'c-', label='Train SC', alpha=0.8)
    if history.get('train_stft_mag'):
        ax.plot(epochs, history['train_stft_mag'], 'm-', label='Train Mag', alpha=0.8)
    if history.get('val_stft_sc'):
        ax.plot(epochs, history['val_stft_sc'], 'c--', label='Val SC', alpha=0.6)
    if history.get('val_stft_mag'):
        ax.plot(epochs, history['val_stft_mag'], 'm--', label='Val Mag', alpha=0.6)
    ax.set_title('STFT Losses (train vs val)')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1, 1]
    if history.get('train_feat_align'):
        ax.plot(epochs, history['train_feat_align'], 'b-', label='Train feat align', alpha=0.8)
    if history.get('val_feat_align'):
        ax.plot(epochs, history['val_feat_align'], 'r--', label='Val feat align', alpha=0.8)
    ax.set_title('Feature Alignment Loss (student vs teacher feat)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 0]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'green', linewidth=2)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    ax = axes[2, 1]
    if (history.get('val_wav_mse') and history.get('val_stft_sc')
            and history.get('val_stft_mag') and history.get('val_mel_loss')):
        val_totals = [
            m + sc + mag + 45.0 * mel
            for m, sc, mag, mel in zip(
                history['val_wav_mse'], history['val_stft_sc'],
                history['val_stft_mag'], history['val_mel_loss'],
            )
        ]
        ax.plot(epochs[:len(val_totals)], val_totals, 'purple', linewidth=2,
                label='val_total')
        best_ep = val_totals.index(min(val_totals))
        ax.axvline(x=best_ep + 1, color='red', ls='--', alpha=0.7,
                   label=f'best ep{best_ep+1}={min(val_totals):.2f}')
        ax.set_title('Val Total Loss (MSE+STFT_SC+STFT_Mag+45×Mel)')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
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
        description='exp_0226a: Encoder LoRA + Feature Alignment Loss (no decoder training)'
    )

    parser.add_argument('--mode', type=str, default='smoke',
                        choices=['smoke', 'epoch'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--encoder_ckpt', type=str, default=EXP0225A_BEST_CKPT,
                        help='exp_0225a best_model_val_total.pt path')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Smaller than 0225a (1e-4) to preserve learned encoder')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:1')

    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    parser.add_argument('--lambda_wav', type=float, default=1.0)
    parser.add_argument('--lambda_stft', type=float, default=1.0)
    parser.add_argument('--lambda_mel', type=float, default=45.0)
    parser.add_argument('--lambda_feat', type=float, default=1.0,
                        help='Feature alignment weight: MSE(student_feat, teacher_feat)')

    parser.add_argument('--stft_fft_sizes', type=str, default='2048,1024,512')
    parser.add_argument('--stft_hop_sizes', type=str, default='512,256,128')
    parser.add_argument('--stft_win_sizes', type=str, default='2048,1024,512')

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

    parser.add_argument('--save_checkpoint_every', type=int, default=10)
    parser.add_argument('--save_audio_interval', type=int, default=25)
    parser.add_argument('--eval_max_batches', type=int, default=30)

    args = parser.parse_args()

    if args.mode == 'smoke':
        args.epochs = max(args.epochs, 5)
        args.eval_max_batches = 5

    fft_sizes = [int(x) for x in args.stft_fft_sizes.split(',')]
    hop_sizes = [int(x) for x in args.stft_hop_sizes.split(',')]
    win_sizes = [int(x) for x in args.stft_win_sizes.split(',')]

    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(f'families/deps/feat_align/runs/enc_feat_align_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0226a_enc_feat_align'
    config['encoder_init'] = f'exp_0225a best_model_val_total.pt: {args.encoder_ckpt}'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0226a: Encoder LoRA + Feature Alignment Loss")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LR: {args.learning_rate} → {args.min_lr}")
    print(f"Encoder LoRA: rank={args.lora_rank}, alpha={args.lora_alpha} (trainable)")
    print(f"VQ: SKIPPED")
    print(f"Decoder: FROZEN")
    print(f"Encoder init: {args.encoder_ckpt}")
    print(f"Loss: λ_wav={args.lambda_wav} + λ_stft={args.lambda_stft} "
          f"+ λ_mel={args.lambda_mel} + λ_feat={args.lambda_feat}")
    print(f"Output: {exp_dir}")
    if log_path:
        print(f"Log: {log_path}")
    print("=" * 70)

    device = torch.device(args.device)
    cuda_preinit(device)

    print("\nLoading data...")
    if args.mode == 'smoke':
        full_ds = AugmentedCurriculumDataset(
            VAL_CACHE, augment=False,
            filter_clean_to_clean=True, compute_snr=False,
        )
        smoke_indices = list(range(min(20, len(full_ds))))
        smoke_ds = torch.utils.data.Subset(full_ds, smoke_indices)
        train_loader = DataLoader(
            smoke_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn_curriculum,
        )
        val_loader = DataLoader(
            smoke_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn_curriculum,
        )
        print(f"Smoke test: {len(smoke_ds)} samples")
    else:
        train_loader = make_train_loader(
            TRAIN_CACHE, batch_size=args.batch_size, num_workers=2,
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

    print("\nBuilding TeacherStudentNoVQ (encoder-only trainable)...")
    model = TeacherStudentNoVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6],
        device=device,
    ).to(device)

    # 載入 exp_0225a encoder LoRA 權重
    ckpt_path = Path(args.encoder_ckpt)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            missing, unexpected = model.load_state_dict(
                ckpt['model_state_dict'], strict=False
            )
            print(f"  Loaded model_state_dict from {ckpt_path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        elif 'encoder_lora_state' in ckpt:
            model.student.load_state_dict(ckpt['encoder_lora_state'], strict=False)
            print(f"  Loaded encoder_lora_state from {ckpt_path}")
        src_ep = ckpt.get('epoch', '?')
        print(f"  Source epoch: {src_ep}")
    else:
        print(f"  WARNING: encoder ckpt not found: {ckpt_path}")
        print(f"  Starting from WavTokenizer pretrained weights")

    mr_stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_sizes=win_sizes,
    ).to(device)
    mel_loss_fn = MelReconstructionLoss(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"\nTrainable params: {trainable_count:,} / {total_params:,} "
          f"({100*trainable_count/total_params:.3f}%)")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.learning_rate,
                   0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if args.use_amp else None

    best_val_total = float('inf')
    best_val_mse = float('inf')
    history = {
        'train_total_loss': [], 'train_wav_mse': [],
        'train_stft_sc': [], 'train_stft_mag': [], 'train_mel': [],
        'train_feat_align': [],
        'val_wav_mse': [], 'val_noisy_mse': [],
        'val_stft_sc': [], 'val_stft_mag': [], 'val_mel_loss': [],
        'val_noisy_stft_sc': [], 'val_noisy_mel': [],
        'val_feat_align': [],
        'lr': [],
    }

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config,
            mr_stft_loss_fn, mel_loss_fn, scaler,
        )
        val_metrics = evaluate(
            model, val_loader, device, config,
            mr_stft_loss_fn, mel_loss_fn, args.eval_max_batches,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_wav_mse'].append(train_metrics['wav_mse'])
        history['train_stft_sc'].append(train_metrics['stft_sc_loss'])
        history['train_stft_mag'].append(train_metrics['stft_mag_loss'])
        history['train_mel'].append(train_metrics['mel_loss'])
        history['train_feat_align'].append(train_metrics['feat_align_loss'])
        history['val_wav_mse'].append(val_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(val_metrics['val_noisy_mse'])
        history['val_stft_sc'].append(val_metrics['val_stft_sc'])
        history['val_stft_mag'].append(val_metrics['val_stft_mag'])
        history['val_mel_loss'].append(val_metrics['val_mel_loss'])
        history['val_noisy_stft_sc'].append(val_metrics['val_noisy_stft_sc'])
        history['val_noisy_mel'].append(val_metrics['val_noisy_mel'])
        history['val_feat_align'].append(val_metrics['val_feat_align'])
        history['lr'].append(current_lr)

        val_mse = val_metrics['val_wav_mse']
        noisy_mse = val_metrics['val_noisy_mse']
        val_total = (val_mse
                     + val_metrics['val_stft_sc']
                     + val_metrics['val_stft_mag']
                     + args.lambda_mel * val_metrics['val_mel_loss'])
        improve_pct = (noisy_mse - val_mse) / noisy_mse * 100 if noisy_mse > 0 else 0

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: total={train_metrics['total_loss']:.4f}  "
              f"wav={train_metrics['wav_mse']:.5f}  "
              f"feat={train_metrics['feat_align_loss']:.5f}  "
              f"mel={train_metrics['mel_loss']:.3f}")
        print(f"  Val:   recon_mse={val_mse:.5f}  noisy_mse={noisy_mse:.5f}  "
              f"val_total={val_total:.4f}  mse_improve=+{improve_pct:.1f}%")
        print(f"         feat_align={val_metrics['val_feat_align']:.5f}  "
              f"stft_sc={val_metrics['val_stft_sc']:.4f}  "
              f"mel={val_metrics['val_mel_loss']:.4f}")
        print(f"  LR={current_lr:.3e}")

        # best_model_val_total.pt（主要 checkpoint 判準）
        if val_total < best_val_total:
            best_val_total = val_total
            torch.save({
                'epoch': epoch,
                'encoder_lora_state': {
                    k: v for k, v in model.student.state_dict().items()
                    if 'lora_' in k
                },
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, exp_dir / 'best_model_val_total.pt')
            print(f"  New best val_total: {best_val_total:.4f} → saved best_model_val_total.pt")

        # best_model.pt（val_wav_mse 判準）
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                'epoch': epoch,
                'encoder_lora_state': {
                    k: v for k, v in model.student.state_dict().items()
                    if 'lora_' in k
                },
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  New best val_mse: {best_val_mse:.5f} → saved best_model.pt")

        if epoch % args.save_checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'encoder_lora_state': {
                    k: v for k, v in model.student.state_dict().items()
                    if 'lora_' in k
                },
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, exp_dir / f'checkpoint_epoch{epoch:03d}.pt')

        if epoch % args.save_audio_interval == 0 or epoch == args.epochs:
            plot_training_curves(history, exp_dir, epoch)
            _save_audio_samples(model, val_loader, device, exp_dir, epoch,
                                num_samples=2, split='val')
            _save_audio_samples(model, train_loader, device, exp_dir, epoch,
                                num_samples=2, split='train')

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_total: {best_val_total:.4f}")
    print(f"Output: {exp_dir}")


if __name__ == '__main__':
    main()

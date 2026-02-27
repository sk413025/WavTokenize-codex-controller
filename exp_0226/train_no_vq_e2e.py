"""
exp_0226: No-VQ End-to-End LoRA（Encoder + Decoder 同時訓練）

架構：
    Encoder LoRA (Trainable, 初始化自 exp_0225a best_model_val_total.pt):
        Noisy Audio → LoRA Encoder → student_encoder_out [B, 512, T]
        （scratch encoder，無 exp_0217 VQ token 預訓練 prior）

    VQ: 完全跳過

    Decoder LoRA (Trainable, WavTokenizer backbone + 新 LoRA):
        student_encoder_out → [backbone LoRA] → head → recon_wav

Loss:
    λ_wav * MSE(recon_wav, clean_wav)
    + λ_stft * MR-STFT(recon_wav, clean_wav)
    + λ_mel * Mel(recon_wav, clean_wav)

設計動機（消融系列）：
    exp_0225a (scratch Encoder only):  PESQ=1.554，無機械音，decoder 天花板
    exp_0225b (scratch Enc frozen + Dec LoRA): 有機械音
    exp_0226 (scratch Enc + Dec E2E):  兩者協同學習，突破各自天花板

    對比 exp_0224 系列（有 exp_0217 prior）：
    exp_0224a (Encoder only): PESQ=1.586
    exp_0226 vs exp_0224 → 量化 E2E 對「有/無 prior」的影響

執行：
    # Smoke test
    python exp_0226/train_no_vq_e2e.py --mode smoke

    # 正式訓練（從 exp_0225a 初始化）
    python exp_0226/train_no_vq_e2e.py --mode epoch --epochs 300 --device cuda:1 \\
        --encoder_ckpt exp_0225/runs/no_vq_scratch_epoch_20260224_032104/best_model_val_total.pt
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0226.models_no_vq_e2e import TeacherStudentNoVQE2E
from exp_0216.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum


# ============================================================
# Constants
# ============================================================

EXP0225A_BEST_CKPT = (
    Path(__file__).parent.parent /
    'exp_0225/runs/no_vq_scratch_epoch_20260224_032104/best_model_val_total.pt'
)

SAMPLE_RATE = 24000


# ============================================================
# Loss Functions
# ============================================================

class STFTLoss(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        ))

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
        if y_hat.dim() == 3: y_hat = y_hat.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)
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
        if y_hat.dim() == 3: y_hat = y_hat.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)
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
            try: s.write(data)
            except Exception: pass

    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass

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
    """E2E 訓練 epoch：Encoder + Decoder LoRA 同時更新"""
    model.train()
    # Teacher 相關部分保持 eval（凍結且無需 dropout）
    model.teacher.feature_extractor.eval()
    # backbone 和 head 需要 train mode 讓 decoder LoRA 的 dropout 生效
    # （decoder_lora_dropout=0.0 所以實際上沒有 dropout，但保持一致）
    model.teacher.backbone.train()
    model.teacher.head.eval()  # head 完全凍結，eval mode
    model.student.train()      # encoder LoRA train mode

    metrics = {
        'total_loss': 0.0,
        'wav_mse': 0.0,
        'stft_sc_loss': 0.0,
        'stft_mag_loss': 0.0,
        'mel_loss': 0.0,
        'nan_batches': 0,
    }
    n_batches = 0
    nan_count = 0
    max_nan_per_epoch = 10

    lambda_wav = config['lambda_wav']
    lambda_stft = config['lambda_stft']
    lambda_mel = config['lambda_mel']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [no-VQ E2E]")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        if clean_audio.dim() == 2: clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2: noisy_audio = noisy_audio.unsqueeze(1)

        if batch_idx % config['grad_accum'] == 0:
            optimizer.zero_grad()

        with autocast(enabled=config['use_amp']):
            out = model.forward_wav(clean_audio, noisy_audio)
            recon_wav = out['recon_wav']

            T = min(clean_audio.shape[-1], recon_wav.shape[-1])
            recon_t = recon_wav[..., :T]
            clean_t = clean_audio[..., :T]

            wav_mse = F.mse_loss(recon_t, clean_t)
            sc_loss, mag_loss = mr_stft_loss_fn(recon_t, clean_t)
            stft_loss = sc_loss + mag_loss
            mel_loss = mel_loss_fn(recon_t, clean_t)

            loss = (
                lambda_wav * wav_mse
                + lambda_stft * stft_loss
                + lambda_mel * mel_loss
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
        n_batches += 1

        pbar.set_postfix({
            'total': f"{loss_val:.4f}",
            'wav': f"{wav_mse.item():.5f}",
            'stft': f"{(sc_loss + mag_loss).item():.3f}",
            'mel': f"{mel_loss.item():.3f}",
        })

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss', 'mel_loss']:
            metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, config,
             mr_stft_loss_fn, mel_loss_fn, max_batches=30) -> Dict:
    model.eval()

    wav_mse_list, noisy_mse_list = [], []
    stft_sc_list, stft_mag_list, mel_list = [], [], []
    noisy_stft_sc_list, noisy_mel_list = [], []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        if clean_audio.dim() == 2: clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2: noisy_audio = noisy_audio.unsqueeze(1)

        out = model.forward_wav(clean_audio, noisy_audio)
        recon_wav = out['recon_wav']

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

    val_total = (
        float(np.mean(wav_mse_list)) +
        float(np.mean(stft_sc_list)) +
        float(np.mean(stft_mag_list)) +
        45.0 * float(np.mean(mel_list))
    ) if wav_mse_list else float('nan')

    model.train()

    return {
        'val_wav_mse': float(np.mean(wav_mse_list)) if wav_mse_list else float('nan'),
        'val_noisy_mse': float(np.mean(noisy_mse_list)) if noisy_mse_list else float('nan'),
        'val_stft_sc': float(np.mean(stft_sc_list)) if stft_sc_list else float('nan'),
        'val_stft_mag': float(np.mean(stft_mag_list)) if stft_mag_list else float('nan'),
        'val_mel_loss': float(np.mean(mel_list)) if mel_list else float('nan'),
        'val_noisy_stft_sc': float(np.mean(noisy_stft_sc_list)) if noisy_stft_sc_list else float('nan'),
        'val_noisy_mel': float(np.mean(noisy_mel_list)) if noisy_mel_list else float('nan'),
        'val_total': val_total,
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

            if clean_audio.dim() == 2: clean_audio = clean_audio.unsqueeze(1)
            if noisy_audio.dim() == 2: noisy_audio = noisy_audio.unsqueeze(1)

            out = model.forward_wav(clean_audio, noisy_audio)
            recon_wav = out['recon_wav']

            B = min(noisy_audio.shape[0], num_samples - saved)
            for b in range(B):
                def _save(tensor, name):
                    wav = tensor[b].squeeze().cpu().float().numpy()
                    wav = np.clip(wav, -1.0, 1.0)
                    wavfile.write(str(audio_dir / name), SAMPLE_RATE,
                                  (wav * 32767).astype(np.int16))

                _save(noisy_audio, f'sample{saved+b:02d}_noisy.wav')
                T = min(clean_audio.shape[-1], recon_wav.shape[-1])
                _save(recon_wav[..., :T], f'sample{saved+b:02d}_recon.wav')
                _save(clean_audio[..., :T], f'sample{saved+b:02d}_clean.wav')

            saved += B

    model.train()
    print(f"  Audio saved ({split}) → {audio_dir}")


def plot_training_curves(history, output_dir, epoch):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'exp_0226: No-VQ E2E LoRA (Epoch {epoch})', fontsize=14)

    epochs = range(1, len(history['train_total_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train total', alpha=0.8)
    ax.set_title('Total Loss (train)')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    if history.get('val_noisy_mse'):
        ax.plot(epochs, history['val_noisy_mse'], 'gray', ls='--',
                label='Noisy baseline', alpha=0.8)
    if history.get('val_wav_mse'):
        ax.plot(epochs, history['val_wav_mse'], 'r-', label='Recon vs Clean', alpha=0.8)
    ax.set_title('Wav MSE (val)')
    ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    if history.get('train_stft_sc'):
        ax.plot(epochs, history['train_stft_sc'], 'c-', label='SC loss', alpha=0.8)
    if history.get('train_stft_mag'):
        ax.plot(epochs, history['train_stft_mag'], 'm-', label='Mag loss', alpha=0.8)
    ax.set_title('STFT Losses (train)')
    ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    if history.get('train_mel'):
        ax.plot(epochs, history['train_mel'], 'orange', label='Train mel', alpha=0.8)
    if history.get('val_mel_loss'):
        ax.plot(epochs, history['val_mel_loss'], 'r-', label='Val mel', alpha=0.8)
    if history.get('val_noisy_mel'):
        ax.plot(epochs, history['val_noisy_mel'], 'gray', ls='--',
                label='Noisy mel', alpha=0.8)
    ax.set_title('Mel Loss')
    ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True)

    ax = axes[2, 0]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'green', linewidth=2)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    ax = axes[2, 1]
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
        description='exp_0226: No-VQ E2E LoRA (Encoder + Decoder 同時訓練)'
    )

    parser.add_argument('--mode', type=str, default='smoke',
                        choices=['smoke', 'epoch'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default=None)

    # Training basics
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='E2E 用較小 lr，避免 encoder 已學到的被破壞')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:1')

    # Encoder LoRA
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    # Decoder LoRA
    parser.add_argument('--decoder_lora_rank', type=int, default=32)
    parser.add_argument('--decoder_lora_alpha', type=int, default=64)

    # Loss weights
    parser.add_argument('--lambda_wav', type=float, default=1.0)
    parser.add_argument('--lambda_stft', type=float, default=1.0)
    parser.add_argument('--lambda_mel', type=float, default=45.0)

    # STFT 設定
    parser.add_argument('--stft_fft_sizes', type=str, default='2048,1024,512')
    parser.add_argument('--stft_hop_sizes', type=str, default='512,256,128')
    parser.add_argument('--stft_win_sizes', type=str, default='2048,1024,512')

    # Encoder checkpoint（從 exp_0225a 初始化）
    parser.add_argument(
        '--encoder_ckpt', type=str,
        default=str(EXP0225A_BEST_CKPT),
        help='exp_0225a best_model_val_total.pt 路徑（scratch encoder）',
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

    if args.mode == 'smoke':
        args.epochs = max(args.epochs, 5)
        args.eval_max_batches = 5

    fft_sizes = [int(x) for x in args.stft_fft_sizes.split(',')]
    hop_sizes = [int(x) for x in args.stft_hop_sizes.split(',')]
    win_sizes = [int(x) for x in args.stft_win_sizes.split(',')]

    # ===== Setup =====
    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(f'exp_0226/runs/no_vq_e2e_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0226_no_vq_e2e_lora'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0226: No-VQ E2E LoRA（Encoder + Decoder 同時訓練）")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LR: {args.learning_rate} → {args.min_lr}  (warmup={args.warmup_epochs})")
    print(f"Encoder LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Decoder LoRA: rank={args.decoder_lora_rank}, alpha={args.decoder_lora_alpha}")
    print(f"VQ: SKIPPED")
    print(f"Encoder init (exp_0225a scratch): {args.encoder_ckpt}")
    print(f"Loss: λ_wav={args.lambda_wav} + λ_stft={args.lambda_stft} * MR-STFT + λ_mel={args.lambda_mel} * mel")
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

    # ===== Model =====
    print("\nBuilding TeacherStudentNoVQE2E...")
    model = TeacherStudentNoVQE2E(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=device,
        decoder_lora_rank=args.decoder_lora_rank,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_lora_dropout=0.0,  # E2E 不用 dropout
    ).to(device)

    # ===== Load Encoder Checkpoint (exp_0225a scratch) =====
    encoder_ckpt_path = Path(args.encoder_ckpt)
    if encoder_ckpt_path.exists():
        print(f"\nLoading encoder checkpoint (exp_0225a scratch): {encoder_ckpt_path}")
        model.load_encoder_checkpoint(str(encoder_ckpt_path))
    else:
        print(f"\n[WARNING] Encoder checkpoint not found: {encoder_ckpt_path}")
        print("  Starting with fresh encoder LoRA (not recommended)")

    # ===== Loss Functions =====
    mr_stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_sizes=win_sizes,
    ).to(device)
    mel_loss_fn = MelReconstructionLoss(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100,
    ).to(device)

    # ===== 驗證 trainable params =====
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"\nTrainable params: {trainable_count:,} / {total_params:,} "
          f"({100*trainable_count/total_params:.3f}%)")

    enc_trainable = sum(
        p.numel() for n, p in model.student.named_parameters() if p.requires_grad
    )
    dec_trainable = sum(
        p.numel() for n, p in model.teacher.backbone.named_parameters() if p.requires_grad
    )
    print(f"  Encoder LoRA: {enc_trainable:,}  |  Decoder LoRA: {dec_trainable:,}")

    # ===== Optimizer + Scheduler =====
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

    # ===== Training Loop =====
    best_val_mse = float('inf')
    best_val_total = float('inf')
    history = {
        'train_total_loss': [], 'train_wav_mse': [],
        'train_stft_sc': [], 'train_stft_mag': [], 'train_mel': [],
        'val_wav_mse': [], 'val_noisy_mse': [], 'val_total': [],
        'val_stft_sc': [], 'val_mel_loss': [],
        'val_noisy_stft_sc': [], 'val_noisy_mel': [],
        'lr': [],
    }

    print("\nStarting E2E training...\n")

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

        # Update history
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_wav_mse'].append(train_metrics['wav_mse'])
        history['train_stft_sc'].append(train_metrics['stft_sc_loss'])
        history['train_stft_mag'].append(train_metrics['stft_mag_loss'])
        history['train_mel'].append(train_metrics['mel_loss'])
        history['val_wav_mse'].append(val_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(val_metrics['val_noisy_mse'])
        history['val_total'].append(val_metrics['val_total'])
        history['val_stft_sc'].append(val_metrics['val_stft_sc'])
        history['val_mel_loss'].append(val_metrics['val_mel_loss'])
        history['val_noisy_stft_sc'].append(val_metrics['val_noisy_stft_sc'])
        history['val_noisy_mel'].append(val_metrics['val_noisy_mel'])
        history['lr'].append(current_lr)

        val_mse = val_metrics['val_wav_mse']
        noisy_mse = val_metrics['val_noisy_mse']
        val_total = val_metrics['val_total']
        improve_pct = (noisy_mse - val_mse) / noisy_mse * 100 if noisy_mse > 0 else 0

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: total={train_metrics['total_loss']:.4f}  "
              f"wav={train_metrics['wav_mse']:.5f}  "
              f"stft={train_metrics['stft_sc_loss']+train_metrics['stft_mag_loss']:.3f}  "
              f"mel={train_metrics['mel_loss']:.3f}")
        print(f"  Val:   recon_mse={val_mse:.5f}  noisy_mse={noisy_mse:.5f}  "
              f"val_total={val_total:.4f}  mse_improve=+{improve_pct:.1f}%")
        print(f"         recon_mel={val_metrics['val_mel_loss']:.4f}  "
              f"noisy_mel={val_metrics['val_noisy_mel']:.4f}  "
              f"stft_sc={val_metrics['val_stft_sc']:.4f}")
        print(f"  LR={current_lr:.3e}")

        # Best model by val_mse
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'val_wav_mse': val_mse,
                    'val_noisy_mse': noisy_mse,
                    'val_total': val_total,
                    'improve_pct': improve_pct,
                },
                'config': config,
            }
            torch.save(best_ckpt, exp_dir / 'best_model.pt')
            print(f"  New best val MSE: {best_val_mse:.5f} → saved best_model.pt")

        # Best model by val_total
        if val_total < best_val_total:
            best_val_total = val_total
            best_ckpt_total = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'val_wav_mse': val_mse,
                    'val_noisy_mse': noisy_mse,
                    'val_total': val_total,
                    'improve_pct': improve_pct,
                },
                'config': config,
            }
            torch.save(best_ckpt_total, exp_dir / 'best_model_val_total.pt')
            print(f"  New best val_total: {best_val_total:.4f} → saved best_model_val_total.pt")

        # Periodic checkpoint
        if epoch % args.save_checkpoint_every == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }
            torch.save(ckpt, exp_dir / f'checkpoint_epoch{epoch:03d}.pt')

        # Periodic audio + plot
        if epoch % args.save_audio_interval == 0 or epoch == args.epochs:
            plot_training_curves(history, exp_dir, epoch)
            _save_audio_samples(model, val_loader, device, exp_dir, epoch,
                                num_samples=2, split='val')
            _save_audio_samples(model, train_loader, device, exp_dir, epoch,
                                num_samples=2, split='train')

        # Save history
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val MSE: {best_val_mse:.5f}, "
          f"Best val_total: {best_val_total:.4f}")
    print(f"Output: {exp_dir}")


if __name__ == '__main__':
    main()

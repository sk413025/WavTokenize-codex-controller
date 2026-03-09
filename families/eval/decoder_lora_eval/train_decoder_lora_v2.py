"""
exp_0223 v2: Decoder LoRA Fine-tune — Multi-Resolution STFT + Mel Loss

改進點（相對 v1 train_decoder_lora.py）：
    v1 使用純 wav-domain MSE，但 MSE 與感知音質（PESQ/STOI）脫鉤。
    v2 新增：
    1. Multi-Resolution STFT Loss（多解析度頻譜損失）
    2. Mel-Spectrogram Reconstruction Loss（梅爾頻譜 L1 損失）
    3. 保留 wav-domain MSE 作為穩定項

    Total loss = λ_wav * wav_mse + λ_stft * mr_stft_loss + λ_mel * mel_loss

    MR-STFT 使用三組 (n_fft, hop, win)，覆蓋不同時頻解析度：
      - (2048, 512, 2048)  — 低頻細節
      - (1024, 256, 1024)  — 中頻平衡
      - (512,  128,  512)  — 高頻瞬態

架構不變：
    - Encoder + VQ: 凍結（繼承 exp_0217 best checkpoint）
    - Decoder backbone ConvNeXt pwconv1/pwconv2: LoRA (rank=32)，可訓練

執行：
    # Smoke test
    python families/eval/decoder_lora_eval/train_decoder_lora_v2.py --mode smoke --epochs 5

    # 正式訓練（繼承 v1 checkpoint）
    python families/eval/decoder_lora_eval/train_decoder_lora_v2.py \\
        --mode epoch --epochs 150 --device cuda:0 \\
        --resume_from families/eval/decoder_lora_eval/runs/decoder_lora_epoch_20260223_010247/best_model.pt

    # 從零開始
    python families/eval/decoder_lora_eval/train_decoder_lora_v2.py \\
        --mode epoch --epochs 150 --device cuda:0

科學目標：
    驗證 Multi-Resolution STFT + Mel Loss 是否能讓 decoder LoRA
    在 wav MSE 之外同時改善 PESQ/STOI 感知指標。
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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
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

SAMPLE_RATE = 24000


# ============================================================
# Multi-Resolution STFT Loss
# ============================================================

class STFTLoss(nn.Module):
    """單一解析度 STFT Loss（spectral convergence + log magnitude）

    Args:
        n_fft: FFT 大小
        hop_length: Hop 大小
        win_length: 窗口大小
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        """初始化 STFTLoss。

        Args:
            n_fft: FFT 點數。
            hop_length: 幀移大小。
            win_length: 窗口長度。
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer(
            'window', torch.hann_window(win_length)
        )

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """計算 STFT 的幅度頻譜。

        Args:
            x: 輸入音訊波形 [B, T]。

        Returns:
            幅度頻譜 [B, F, T']。
        """
        # x: [B, T]
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        )
        return torch.abs(spec)  # magnitude: [B, F, T']

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算頻譜收斂損失與對數幅度損失。

        Args:
            y_hat: 預測音訊 [B, T]。
            y: 真實音訊 [B, T]。

        Returns:
            (spectral_convergence_loss, log_magnitude_loss) 的元組。
        """
        mag_hat = self._stft(y_hat)
        mag = self._stft(y)

        # Spectral convergence: Frobenius norm ratio
        sc_loss = torch.norm(mag - mag_hat, p='fro') / (torch.norm(mag, p='fro') + 1e-7)

        # Log magnitude: L1 distance in log domain
        log_mag_loss = F.l1_loss(
            torch.log(mag.clamp(min=1e-7)),
            torch.log(mag_hat.clamp(min=1e-7)),
        )

        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """多解析度 STFT 損失函數

    使用多組 (n_fft, hop, win) 捕捉不同時頻尺度的頻譜特性。

    Args:
        fft_sizes: FFT 大小列表。
        hop_sizes: Hop 大小列表。
        win_sizes: 窗口大小列表。
    """

    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512],
        hop_sizes: List[int] = [512, 256, 128],
        win_sizes: List[int] = [2048, 1024, 512],
    ):
        """初始化 MultiResolutionSTFTLoss。

        Args:
            fft_sizes: 各解析度的 FFT 大小列表。
            hop_sizes: 各解析度的 hop 大小列表。
            win_sizes: 各解析度的窗口大小列表。
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft, hop, win)
            for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算多解析度 STFT 損失。

        Args:
            y_hat: 預測音訊 [B, T] 或 [B, 1, T]。
            y: 真實音訊 [B, T] 或 [B, 1, T]。

        Returns:
            (total_sc_loss, total_log_mag_loss) 的元組，
            各為所有解析度的平均值。
        """
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)

        sc_loss = 0.0
        mag_loss = 0.0
        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(y_hat, y)
            sc_loss += sc
            mag_loss += mag

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class MelReconstructionLoss(nn.Module):
    """Mel-Spectrogram 重建損失

    使用 torchaudio 計算 Mel 頻譜的 L1 距離（log domain）。

    Args:
        sample_rate: 採樣率。
        n_fft: FFT 大小。
        hop_length: Hop 大小。
        n_mels: Mel 濾波器數量。
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
    ):
        """初始化 MelReconstructionLoss。

        Args:
            sample_rate: 音訊採樣率。
            n_fft: FFT 大小。
            hop_length: Hop 大小。
            n_mels: Mel 濾波器組數量。
        """
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """計算 Mel 頻譜重建損失。

        Args:
            y_hat: 預測音訊 [B, 1, T] 或 [B, T]。
            y: 真實音訊 [B, 1, T] 或 [B, T]。

        Returns:
            Mel 頻譜 L1 損失值。
        """
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
    """設定日誌輸出，同時寫入檔案與 stdout/stderr。

    Args:
        output_dir: 日誌輸出目錄。

    Returns:
        日誌檔案路徑。
    """
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
    """設定所有隨機種子以確保可重現性。

    Args:
        seed: 隨機種子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Fixed seed={seed}")


def cuda_preinit(device: torch.device, retries: int = 10, sleep_s: float = 2.0):
    """預初始化 CUDA 裝置，含重試機制。

    Args:
        device: CUDA 裝置。
        retries: 最大重試次數。
        sleep_s: 每次重試間隔秒數。
    """
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
    """建立驗證集 DataLoader。

    Args:
        val_cache_path: 驗證集快取路徑。
        batch_size: 批次大小。
        num_workers: 資料載入工作者數。

    Returns:
        驗證集 DataLoader。
    """
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
    """建立訓練集 DataLoader（含資料增強）。

    Args:
        train_cache_path: 訓練集快取路徑。
        batch_size: 批次大小。
        num_workers: 資料載入工作者數。
        snr_remix_prob: SNR 混音增強機率。
        snr_remix_range: SNR 混音範圍 (min, max) dB。
        random_gain_prob: 隨機增益機率。
        random_gain_db: 隨機增益最大 dB。
        random_crop_prob: 隨機裁切機率。
        random_crop_min_ratio: 隨機裁切最小比例。
        time_stretch_prob: 時間伸縮機率。
        time_stretch_range: 時間伸縮範圍 (min, max)。

    Returns:
        訓練集 DataLoader。
    """
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
                mr_stft_loss_fn, mel_loss_fn, scaler=None) -> Dict:
    """Decoder LoRA v2 訓練 epoch

    loss = λ_wav * MSE(recon, clean)
         + λ_stft * (sc_loss + log_mag_loss)
         + λ_mel * mel_loss

    Args:
        model: TeacherStudentDecoderLoRA 模型。
        dataloader: 訓練資料載入器。
        optimizer: 最佳化器。
        device: 計算裝置。
        epoch: 當前 epoch 編號。
        config: 訓練設定字典。
        mr_stft_loss_fn: Multi-Resolution STFT Loss 函數。
        mel_loss_fn: Mel Spectrogram Loss 函數。
        scaler: GradScaler（AMP 使用）。

    Returns:
        包含各項 loss 指標的字典。
    """
    model.train()
    model.teacher.backbone.train()  # LoRA dropout active
    model.teacher.head.eval()

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

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [decoder LoRA v2]")

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

            # Align length
            T_clean = clean_audio.shape[-1]
            T_recon = recon_wav.shape[-1]
            T = min(T_clean, T_recon)
            recon_t = recon_wav[..., :T]
            clean_t = clean_audio[..., :T]

            # 1) Wav MSE
            wav_mse = F.mse_loss(recon_t, clean_t)

            # 2) Multi-Resolution STFT Loss
            sc_loss, mag_loss = mr_stft_loss_fn(recon_t, clean_t)
            stft_loss = sc_loss + mag_loss

            # 3) Mel Loss
            mel_loss = mel_loss_fn(recon_t, clean_t)

            # Combined loss
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
def evaluate_decoder_v2(model, dataloader, device, config,
                        mr_stft_loss_fn, mel_loss_fn,
                        max_batches=30) -> Dict:
    """Decoder LoRA v2 評估

    計算 val 集上的各種 loss 分量，以及相對於 noisy 的改善。

    Args:
        model: TeacherStudentDecoderLoRA 模型。
        dataloader: 驗證資料載入器。
        device: 計算裝置。
        config: 訓練設定字典。
        mr_stft_loss_fn: Multi-Resolution STFT Loss 函數。
        mel_loss_fn: Mel Spectrogram Loss 函數。
        max_batches: 最大評估批次數。

    Returns:
        包含各項 loss 指標的字典。
    """
    model.eval()

    wav_mse_list = []
    noisy_mse_list = []
    stft_sc_list = []
    stft_mag_list = []
    mel_list = []
    # Also eval noisy→clean spectral losses for comparison
    noisy_stft_sc_list = []
    noisy_mel_list = []

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

        # Recon metrics
        wav_mse_list.append(F.mse_loss(recon_t, clean_t).item())
        sc, mag = mr_stft_loss_fn(recon_t, clean_t)
        stft_sc_list.append(sc.item())
        stft_mag_list.append(mag.item())
        mel_list.append(mel_loss_fn(recon_t, clean_t).item())

        # Noisy baseline metrics
        noisy_mse_list.append(F.mse_loss(noisy_t, clean_t).item())
        sc_n, _ = mr_stft_loss_fn(noisy_t, clean_t)
        noisy_stft_sc_list.append(sc_n.item())
        noisy_mel_list.append(mel_loss_fn(noisy_t, clean_t).item())

    model.train()

    return {
        'val_wav_mse': float(np.mean(wav_mse_list)) if wav_mse_list else float('nan'),
        'val_noisy_mse': float(np.mean(noisy_mse_list)) if noisy_mse_list else float('nan'),
        'val_stft_sc': float(np.mean(stft_sc_list)) if stft_sc_list else float('nan'),
        'val_stft_mag': float(np.mean(stft_mag_list)) if stft_mag_list else float('nan'),
        'val_mel_loss': float(np.mean(mel_list)) if mel_list else float('nan'),
        'val_noisy_stft_sc': float(np.mean(noisy_stft_sc_list)) if noisy_stft_sc_list else float('nan'),
        'val_noisy_mel': float(np.mean(noisy_mel_list)) if noisy_mel_list else float('nan'),
    }


def plot_training_curves_v2(history, output_dir, epoch):
    """繪製 v2 訓練曲線（多面板含各 loss 分量）。

    Args:
        history: 訓練歷史字典。
        output_dir: 圖表輸出目錄。
        epoch: 當前 epoch 編號。
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'exp_0223 v2: Decoder LoRA + MR-STFT + Mel (Epoch {epoch})',
                 fontsize=14)

    epochs = range(1, len(history['train_total_loss']) + 1)

    # (0,0) Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train total', alpha=0.8)
    ax.set_title('Total Loss (train)')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    # (0,1) Wav MSE: recon vs noisy comparison
    ax = axes[0, 1]
    if history.get('val_noisy_mse'):
        ax.plot(epochs, history['val_noisy_mse'], 'gray', ls='--',
                label='Noisy vs Clean (baseline)', alpha=0.8)
    if history.get('val_wav_mse'):
        ax.plot(epochs, history['val_wav_mse'], 'r-', label='Recon vs Clean', alpha=0.8)
    ax.set_title('Wav MSE (val)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # (1,0) STFT losses
    ax = axes[1, 0]
    if history.get('train_stft_sc'):
        ax.plot(epochs, history['train_stft_sc'], 'c-', label='SC loss', alpha=0.8)
    if history.get('train_stft_mag'):
        ax.plot(epochs, history['train_stft_mag'], 'm-', label='Mag loss', alpha=0.8)
    ax.set_title('STFT Losses (train)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # (1,1) Mel loss
    ax = axes[1, 1]
    if history.get('train_mel'):
        ax.plot(epochs, history['train_mel'], 'orange', label='Train mel', alpha=0.8)
    if history.get('val_mel_loss'):
        ax.plot(epochs, history['val_mel_loss'], 'r-', label='Val mel', alpha=0.8)
    if history.get('val_noisy_mel'):
        ax.plot(epochs, history['val_noisy_mel'], 'gray', ls='--',
                label='Noisy mel (baseline)', alpha=0.8)
    ax.set_title('Mel Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # (2,0) Learning rate
    ax = axes[2, 0]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'green', linewidth=2)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    # (2,1) Improvement %
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
    """主訓練流程。"""
    parser = argparse.ArgumentParser(
        description='exp_0223 v2: Decoder LoRA + MR-STFT + Mel Loss'
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
    parser.add_argument('--device', type=str, default='cuda:0')

    # Encoder LoRA（用於 checkpoint 載入，不訓練）
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    # Decoder LoRA
    parser.add_argument('--decoder_lora_rank', type=int, default=32)
    parser.add_argument('--decoder_lora_alpha', type=int, default=64)
    parser.add_argument('--decoder_lora_dropout', type=float, default=0.1)

    # Loss weights
    parser.add_argument('--lambda_wav', type=float, default=1.0,
                        help='Wav MSE loss 權重')
    parser.add_argument('--lambda_stft', type=float, default=1.0,
                        help='Multi-Resolution STFT loss 權重')
    parser.add_argument('--lambda_mel', type=float, default=45.0,
                        help='Mel spectrogram loss 權重 (WavTokenizer 原始使用 45)')

    # STFT loss 設定
    parser.add_argument('--stft_fft_sizes', type=str, default='2048,1024,512',
                        help='MR-STFT FFT 大小（逗號分隔）')
    parser.add_argument('--stft_hop_sizes', type=str, default='512,256,128',
                        help='MR-STFT hop 大小（逗號分隔）')
    parser.add_argument('--stft_win_sizes', type=str, default='2048,1024,512',
                        help='MR-STFT 窗口大小（逗號分隔）')

    # Encoder checkpoint
    parser.add_argument(
        '--encoder_ckpt', type=str,
        default=str(EXP0217_BEST_CKPT),
        help='exp_0217 best_model.pt 路徑',
    )

    # Resume from v1 or v2 checkpoint
    parser.add_argument(
        '--resume_from', type=str, default=None,
        help='從先前 decoder LoRA checkpoint 繼續（best_model.pt 或 checkpoint_epochXXX.pt）',
    )
    parser.add_argument(
        '--resume_epoch', type=int, default=0,
        help='繼續訓練的起始 epoch（若 resume_from 有值會自動偵測）',
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

    # Smoke test: override
    if args.mode == 'smoke':
        args.epochs = max(args.epochs, 5)
        args.eval_max_batches = 5

    # Parse STFT sizes
    fft_sizes = [int(x) for x in args.stft_fft_sizes.split(',')]
    hop_sizes = [int(x) for x in args.stft_hop_sizes.split(',')]
    win_sizes = [int(x) for x in args.stft_win_sizes.split(',')]

    # ===== Setup =====
    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(f'families/eval/decoder_lora_eval/runs/decoder_lora_v2_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0223_decoder_lora_v2'
    config['loss_description'] = (
        f"λ_wav={args.lambda_wav} * wav_mse "
        f"+ λ_stft={args.lambda_stft} * MR-STFT "
        f"+ λ_mel={args.lambda_mel} * mel_loss"
    )

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0223 v2: Decoder LoRA + MR-STFT + Mel Loss")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed} (fixed)")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LR: {args.learning_rate} → {args.min_lr}")
    print(f"Decoder LoRA: rank={args.decoder_lora_rank}, alpha={args.decoder_lora_alpha}")
    print(f"Encoder LoRA: rank={args.lora_rank} (frozen)")
    print(f"Encoder ckpt: {args.encoder_ckpt}")
    print(f"Loss: {config['loss_description']}")
    print(f"  MR-STFT FFT sizes: {fft_sizes}")
    print(f"  MR-STFT hop sizes: {hop_sizes}")
    print(f"  MR-STFT win sizes: {win_sizes}")
    if args.resume_from:
        print(f"Resume from: {args.resume_from}")
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
        print("  Starting with fresh encoder LoRA (not recommended)")

    # ===== Loss Functions =====
    print("\nInitializing loss functions...")
    mr_stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=hop_sizes,
        win_sizes=win_sizes,
    ).to(device)
    print(f"  MR-STFT Loss: {len(fft_sizes)} resolutions")

    mel_loss_fn = MelReconstructionLoss(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
    ).to(device)
    print(f"  Mel Loss: n_fft=1024, hop=256, n_mels=100, sr={SAMPLE_RATE}")

    print(f"  Loss weights: wav={args.lambda_wav}, stft={args.lambda_stft}, mel={args.lambda_mel}")

    # ===== 驗證 trainable params =====
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"\nTrainable params: {trainable_count:,} / {total_params:,} "
          f"({100*trainable_count/total_params:.3f}%)")

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
        """餘弦退火學習率排程。

        Args:
            epoch: 當前 epoch。

        Returns:
            學習率乘數。
        """
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.learning_rate,
                   0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if args.use_amp else None

    # ===== Resume =====
    start_epoch = 1
    best_val_mse = float('inf')

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"\nResuming decoder LoRA from: {resume_path}")
            ckpt = torch.load(str(resume_path), map_location='cpu', weights_only=False)

            # Load decoder LoRA weights
            if 'decoder_lora_state' in ckpt:
                # Load only decoder LoRA params
                lora_state = ckpt['decoder_lora_state']
                current_state = model.state_dict()
                for k, v in lora_state.items():
                    if k in current_state:
                        current_state[k] = v
                model.load_state_dict(current_state, strict=False)
                print(f"  Decoder LoRA state loaded ({len(lora_state)} params)")
            elif 'model_state_dict' in ckpt:
                # Load full model, filter to decoder LoRA only
                full_state = ckpt['model_state_dict']
                lora_keys = [k for k in full_state if 'lora_' in k and 'backbone' in k]
                if lora_keys:
                    current_state = model.state_dict()
                    loaded = 0
                    for k in lora_keys:
                        if k in current_state:
                            current_state[k] = full_state[k]
                            loaded += 1
                    model.load_state_dict(current_state, strict=False)
                    print(f"  Decoder LoRA loaded from model_state_dict ({loaded} keys)")

            # Resume epoch
            if 'epoch' in ckpt:
                if args.resume_epoch > 0:
                    start_epoch = args.resume_epoch + 1
                else:
                    start_epoch = ckpt['epoch'] + 1
                print(f"  Resuming from epoch {start_epoch}")

            # Resume optimizer
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print("  Optimizer state restored")
                except Exception as e:
                    print(f"  Optimizer state not restored: {e}")

            # Resume scheduler
            if 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    print("  Scheduler state restored")
                except Exception as e:
                    print(f"  Scheduler state not restored: {e}")

            # Resume best MSE
            if 'metrics' in ckpt and isinstance(ckpt['metrics'], dict):
                best_val_mse = ckpt['metrics'].get('val_wav_mse', float('inf'))
                print(f"  Best val MSE from checkpoint: {best_val_mse:.5f}")
        else:
            print(f"\n[WARNING] Resume checkpoint not found: {resume_path}")

    # ===== Training History =====
    history = {
        'train_total_loss': [],
        'train_wav_mse': [],
        'train_stft_sc': [],
        'train_stft_mag': [],
        'train_mel': [],
        'val_wav_mse': [],
        'val_noisy_mse': [],
        'val_stft_sc': [],
        'val_stft_mag': [],
        'val_mel_loss': [],
        'val_noisy_mel': [],
        'lr': [],
    }

    # ===== Training Loop =====
    print(f"\nStarting training (epoch {start_epoch} → {args.epochs})...")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            config=config, mr_stft_loss_fn=mr_stft_loss_fn,
            mel_loss_fn=mel_loss_fn, scaler=scaler,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_metrics = evaluate_decoder_v2(
            model, val_loader, device, config,
            mr_stft_loss_fn=mr_stft_loss_fn,
            mel_loss_fn=mel_loss_fn,
            max_batches=args.eval_max_batches,
        )

        epoch_time = time.time() - epoch_start

        # Update history
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_wav_mse'].append(train_metrics['wav_mse'])
        history['train_stft_sc'].append(train_metrics['stft_sc_loss'])
        history['train_stft_mag'].append(train_metrics['stft_mag_loss'])
        history['train_mel'].append(train_metrics['mel_loss'])
        history['val_wav_mse'].append(val_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(val_metrics['val_noisy_mse'])
        history['val_stft_sc'].append(val_metrics['val_stft_sc'])
        history['val_stft_mag'].append(val_metrics['val_stft_mag'])
        history['val_mel_loss'].append(val_metrics['val_mel_loss'])
        history['val_noisy_mel'].append(val_metrics['val_noisy_mel'])
        history['lr'].append(current_lr)

        # Improvement
        noisy_mse = val_metrics['val_noisy_mse']
        recon_mse = val_metrics['val_wav_mse']
        improvement_pct = (noisy_mse - recon_mse) / noisy_mse * 100 if noisy_mse > 0 else 0.0

        noisy_mel = val_metrics['val_noisy_mel']
        recon_mel = val_metrics['val_mel_loss']
        mel_improvement = (noisy_mel - recon_mel) / noisy_mel * 100 if noisy_mel > 0 else 0.0

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: total={train_metrics['total_loss']:.4f}  "
              f"wav={train_metrics['wav_mse']:.5f}  "
              f"stft_sc={train_metrics['stft_sc_loss']:.4f}  "
              f"stft_mag={train_metrics['stft_mag_loss']:.4f}  "
              f"mel={train_metrics['mel_loss']:.4f}")
        print(f"  Val:   recon_mse={recon_mse:.5f}  noisy_mse={noisy_mse:.5f}  "
              f"mse_improve={improvement_pct:+.1f}%")
        print(f"         recon_mel={recon_mel:.4f}  noisy_mel={noisy_mel:.4f}  "
              f"mel_improve={mel_improvement:+.1f}%")
        print(f"         stft_sc={val_metrics['val_stft_sc']:.4f}  "
              f"stft_mag={val_metrics['val_stft_mag']:.4f}")
        print(f"  LR={current_lr:.2e}")

        # Save best model (based on val wav MSE for consistency with v1)
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
        if epoch % args.save_audio_interval == 0 or epoch == start_epoch or epoch == args.epochs:
            try:
                plot_training_curves_v2(history, exp_dir, epoch)
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
        plot_training_curves_v2(history, exp_dir, args.epochs)
    except Exception as e:
        print(f"  Warning: Final plot failed: {e}")

    summary = {
        'experiment': 'exp_0223_decoder_lora_v2',
        'mode': args.mode,
        'total_epochs': args.epochs,
        'start_epoch': start_epoch,
        'seed': args.seed,
        'best_val_mse': best_val_mse,
        'loss_weights': {
            'lambda_wav': args.lambda_wav,
            'lambda_stft': args.lambda_stft,
            'lambda_mel': args.lambda_mel,
        },
        'stft_config': {
            'fft_sizes': fft_sizes,
            'hop_sizes': hop_sizes,
            'win_sizes': win_sizes,
        },
        'config': config,
        'final_metrics': {
            'train_total_loss': history['train_total_loss'][-1],
            'train_wav_mse': history['train_wav_mse'][-1],
            'val_wav_mse': history['val_wav_mse'][-1],
            'val_noisy_mse': history['val_noisy_mse'][-1],
            'val_mel_loss': history['val_mel_loss'][-1],
            'val_noisy_mel': history['val_noisy_mel'][-1],
        },
        'baseline_reference': {
            'noisy_teacher_vq_pesq': 1.203,
            'clean_teacher_vq_pesq': 1.790,
            'exp_0217_pesq': 1.147,
            'exp_0217_stoi': 0.511,
        },
        'v1_comparison': {
            'v1_best_val_mse': 0.01542,
            'v1_description': 'wav-domain MSE only',
        },
    }
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"exp_0223 v2 training done!")
    print(f"  Best val MSE: {best_val_mse:.5f}")
    print(f"  Results: {exp_dir}")
    print(f"{'='*70}")


def _save_audio_samples(model, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    """儲存 noisy / clean / recon 音檔。

    Args:
        model: 模型。
        loader: 資料載入器。
        device: 計算裝置。
        output_dir: 輸出目錄。
        epoch: 當前 epoch。
        num_samples: 儲存樣本數。
        split: 資料集分割名稱（train/val）。
    """
    try:
        import scipy.io.wavfile as wav
        import numpy as np
    except ImportError:
        return

    model.eval()
    audio_dir = output_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    SR = SAMPLE_RATE
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
            """將 tensor 轉換為 16-bit PCM numpy array。

            Args:
                t: 音訊 tensor。

            Returns:
                16-bit int numpy array。
            """
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

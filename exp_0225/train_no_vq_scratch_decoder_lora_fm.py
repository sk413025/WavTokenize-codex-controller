"""
exp_0225d: No-VQ + Decoder LoRA + Frozen Discriminator Feature Matching Loss

架構：
    Encoder (Frozen, 繼承 exp_0225a):
        Noisy Audio → LoRA Encoder → student_encoder_out (連續 [B, 512, T])

    VQ: 完全跳過

    Decoder LoRA (Trainable):
        student_encoder_out → [backbone LoRA pwconv1/pwconv2] → head → recon_wav

    Discriminators (Frozen, 載入 WavTokenizer 官方 checkpoint):
        - MultiPeriodDiscriminator (MPD): periods = [2, 3, 5, 7, 11]
        - MultiResolutionDiscriminator (MRD): resolutions = [(1024,256,1024), (2048,512,2048), (512,128,512)]
        - DACDiscriminator: MPD + MRD (DAC 版)
        僅向 forward 取 feature maps，不做 adversarial training，不更新。

Loss:
    λ_wav * MSE(recon_wav, clean_wav)
    + λ_stft * MR-STFT(recon_wav, clean_wav)
    + λ_mel * Mel(recon_wav, clean_wav)
    + λ_fm * FeatureMatchingLoss(disc_fmaps_recon, disc_fmaps_clean)    ← 新增

Feature Matching Loss 設計動機：
    原始 WavTokenizer 使用 3 組 discriminator 做 GAN 訓練，其中 feature matching loss
    捕捉了多尺度的 temporal pattern、phase coherence、harmonic structure。

    完整 GAN 訓練風險高（mode collapse、需要雙 optimizer）。
    Frozen Discriminator FM 是一個折衷方案：
    - 載入官方預訓練 disc 權重（在大量乾淨語音上訓練過）
    - 凍結所有 disc 參數（不做 adversarial training）
    - 只用 disc 的中間 feature maps 做 L1 loss（perceptual loss 概念）
    - 單一 optimizer，訓練穩定

    這相當於把 discriminator 當成 pre-trained perceptual feature extractor，
    類似於 VGG perceptual loss 在影像重建中的角色。

消融對照：
    exp_0225b: MSE + MR-STFT + Mel                     → 基線（機械感）
    exp_0225c: MSE + MR-STFT + Mel + Phase Loss         → 直接監督相位
    exp_0225d: MSE + MR-STFT + Mel + Frozen Disc FM     → 本腳本（perceptual loss）

執行方式：
    python exp_0225/train_no_vq_scratch_decoder_lora_fm.py \\
        --mode epoch --epochs 300 --device cuda:0 \\
        --encoder_ckpt exp_0225/runs/no_vq_scratch_epoch_YYYYMMDD_HHMMSS/best_model_val_total.pt \\
        --lambda_fm 2.0

    # Smoke test
    python exp_0225/train_no_vq_scratch_decoder_lora_fm.py --mode smoke
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0224.models_no_vq_decoder_lora import TeacherStudentNoVQDecoderLoRA
from exp_0216.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum
from decoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from decoder.discriminator_dac import DACDiscriminator


# ============================================================
# Constants
# ============================================================

EXP0225A_BEST_CKPT_GLOB = 'exp_0225/runs/no_vq_scratch_epoch_*/best_model_val_total.pt'

EXP0217_BEST_CKPT = (
    Path(__file__).parent.parent /
    'exp_0217/runs/t453_weighted_epoch_20260217_104843/best_model.pt'
)

# 官方 WavTokenizer checkpoint（含 discriminator 權重）
WAVTOK_FULL_CKPT = '/home/sbplab/ruizi/WavTokenizer-main/wavtokenizer_large_speech_320_24k.ckpt'

SAMPLE_RATE = 24000


# ============================================================
# Loss Functions
# ============================================================

class STFTLoss(nn.Module):
    """單一解析度 STFT Loss。

    Args:
        n_fft: FFT 大小
        hop_length: hop 大小
        win_length: window 大小
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """計算 STFT magnitude。

        Args:
            x: 音訊波形 [B, T]

        Returns:
            STFT magnitude [B, F, T']
        """
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        )
        return torch.abs(spec)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算 spectral convergence 和 log magnitude loss。

        Args:
            y_hat: 重建波形
            y: 目標波形

        Returns:
            (spectral_convergence, log_magnitude_loss)
        """
        mag_hat = self._stft(y_hat)
        mag = self._stft(y)
        sc_loss = torch.norm(mag - mag_hat, p='fro') / (torch.norm(mag, p='fro') + 1e-7)
        log_mag_loss = F.l1_loss(
            torch.log(mag.clamp(min=1e-7)),
            torch.log(mag_hat.clamp(min=1e-7)),
        )
        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """多解析度 STFT Loss。

    Args:
        fft_sizes: FFT 大小列表
        hop_sizes: hop 大小列表
        win_sizes: window 大小列表
    """

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
        """計算多解析度平均 STFT loss。

        Args:
            y_hat: 重建波形
            y: 目標波形

        Returns:
            (avg_spectral_convergence, avg_log_magnitude_loss)
        """
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
    """Mel 頻譜重建 Loss。

    Args:
        sample_rate: 取樣率
        n_fft: FFT 大小
        hop_length: hop 大小
        n_mels: Mel 頻帶數
    """

    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, center=True, power=1,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """計算 Mel 頻譜 L1 Loss。

        Args:
            y_hat: 重建波形
            y: 目標波形

        Returns:
            Mel L1 loss
        """
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        mel_hat = torch.log(self.mel_spec(y_hat).clamp(min=1e-7))
        mel = torch.log(self.mel_spec(y).clamp(min=1e-7))
        return F.l1_loss(mel, mel_hat)


class FrozenDiscriminatorFMLoss(nn.Module):
    """Frozen Discriminator Feature Matching Loss。

    載入 WavTokenizer 官方預訓練的 3 組 discriminator，凍結所有參數，
    僅用它們的中間 feature maps 計算 L1 loss（perceptual loss）。

    不做 adversarial training，不需要雙 optimizer，訓練穩定。

    Args:
        wavtok_ckpt_path: WavTokenizer 完整 checkpoint 路徑（含 discriminator 權重）
        use_mpd: 是否使用 MultiPeriodDiscriminator
        use_mrd: 是否使用 MultiResolutionDiscriminator
        use_dac: 是否使用 DACDiscriminator
    """

    def __init__(
        self,
        wavtok_ckpt_path: str,
        use_mpd: bool = True,
        use_mrd: bool = True,
        use_dac: bool = True,
    ):
        super().__init__()

        self.use_mpd = use_mpd
        self.use_mrd = use_mrd
        self.use_dac = use_dac

        # 建立 discriminators
        if use_mpd:
            self.mpd = MultiPeriodDiscriminator()
        if use_mrd:
            self.mrd = MultiResolutionDiscriminator()
        if use_dac:
            self.dac = DACDiscriminator()

        # 從 WavTokenizer checkpoint 載入權重
        self._load_disc_weights(wavtok_ckpt_path)

        # 凍結所有 discriminator 參數
        for p in self.parameters():
            p.requires_grad_(False)

        total_disc_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"FrozenDiscriminatorFMLoss 初始化完成:")
        print(f"  MPD: {'✓' if use_mpd else '✗'}")
        print(f"  MRD: {'✓' if use_mrd else '✗'}")
        print(f"  DAC: {'✓' if use_dac else '✗'}")
        print(f"  Total disc params: {total_disc_params:,} (全部凍結)")
        print(f"{'='*60}\n")

    def _load_disc_weights(self, ckpt_path: str):
        """從 WavTokenizer checkpoint 載入 discriminator 權重。

        Args:
            ckpt_path: checkpoint 路徑
        """
        print(f"Loading discriminator weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        if self.use_mpd:
            mpd_state = {
                k.replace('multiperioddisc.', ''): v
                for k, v in state_dict.items()
                if k.startswith('multiperioddisc.')
            }
            missing, unexpected = self.mpd.load_state_dict(mpd_state, strict=False)
            print(f"  MPD loaded: {len(mpd_state)} keys "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")

        if self.use_mrd:
            mrd_state = {
                k.replace('multiresddisc.', ''): v
                for k, v in state_dict.items()
                if k.startswith('multiresddisc.')
            }
            missing, unexpected = self.mrd.load_state_dict(mrd_state, strict=False)
            print(f"  MRD loaded: {len(mrd_state)} keys "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")

        if self.use_dac:
            dac_state = {
                k.replace('dac.', ''): v
                for k, v in state_dict.items()
                if k.startswith('dac.')
            }
            missing, unexpected = self.dac.load_state_dict(dac_state, strict=False)
            print(f"  DAC loaded: {len(dac_state)} keys "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")

    def _feature_matching_loss(
        self, fmap_real: List[List[torch.Tensor]],
        fmap_gen: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """計算 feature matching L1 loss。

        Args:
            fmap_real: 真實音訊的 feature maps
            fmap_gen: 生成音訊的 feature maps

        Returns:
            feature matching loss
        """
        loss = 0.0
        count = 0
        for dr, dg in zip(fmap_real, fmap_gen):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl.detach() - gl))
                count += 1
        return loss / max(count, 1)

    def _dac_feature_matching_loss(
        self, fmaps_real: List, fmaps_gen: List
    ) -> torch.Tensor:
        """計算 DAC discriminator 的 feature matching loss。

        DAC discriminator 的輸出格式不同於 MPD/MRD，
        每個 sub-disc 回傳一個 list of feature maps。

        Args:
            fmaps_real: 真實音訊的 DAC feature maps
            fmaps_gen: 生成音訊的 DAC feature maps

        Returns:
            feature matching loss
        """
        loss = 0.0
        count = 0
        for disc_real, disc_gen in zip(fmaps_real, fmaps_gen):
            # 每個 sub-disc 的結果是 feature map list
            # 最後一個是 final output，前面的是 intermediate features
            for rl, gl in zip(disc_real[:-1], disc_gen[:-1]):
                loss += torch.mean(torch.abs(rl.detach() - gl))
                count += 1
        return loss / max(count, 1)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """計算 frozen discriminator feature matching loss。

        Args:
            y_hat: 重建波形 [B, 1, T] 或 [B, T]
            y: 目標波形 [B, 1, T] 或 [B, T]

        Returns:
            feature matching loss (scalar)
        """
        if y_hat.dim() == 3:
            y_hat_1d = y_hat.squeeze(1)
        else:
            y_hat_1d = y_hat
        if y.dim() == 3:
            y_1d = y.squeeze(1)
        else:
            y_1d = y

        total_fm_loss = 0.0
        n_discs = 0

        # Disc forward 全部在 no_grad 下執行（disc 已凍結），
        # 只保留 gen feature maps 的計算圖讓 gradient 流回 decoder LoRA
        if self.use_mpd:
            with torch.no_grad():
                _, _, fmap_r_mp, _ = self.mpd(y=y_1d, y_hat=y_hat_1d)
            # gen 部分需要 gradient，重新跑一次
            _, _, _, fmap_g_mp = self.mpd(y=y_1d, y_hat=y_hat_1d)
            fm_mp = self._feature_matching_loss(fmap_r_mp, fmap_g_mp)
            total_fm_loss += fm_mp
            n_discs += 1

        if self.use_mrd:
            with torch.no_grad():
                _, _, fmap_r_mrd, _ = self.mrd(y=y_1d, y_hat=y_hat_1d)
            _, _, _, fmap_g_mrd = self.mrd(y=y_1d, y_hat=y_hat_1d)
            fm_mrd = self._feature_matching_loss(fmap_r_mrd, fmap_g_mrd)
            total_fm_loss += fm_mrd
            n_discs += 1

        if self.use_dac:
            # DAC input 需要 [B, 1, T] 格式
            if y_hat.dim() == 2:
                y_hat_2d = y_hat.unsqueeze(1)
            else:
                y_hat_2d = y_hat
            if y.dim() == 2:
                y_2d = y.unsqueeze(1)
            else:
                y_2d = y
            with torch.no_grad():
                fmaps_real = self.dac(y_2d)
            fmaps_gen = self.dac(y_hat_2d)
            fm_dac = self._dac_feature_matching_loss(fmaps_real, fmaps_gen)
            total_fm_loss += fm_dac
            n_discs += 1

        return total_fm_loss / max(n_discs, 1)


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
    """設定日誌輸出到檔案和 stdout。

    Args:
        output_dir: 輸出目錄

    Returns:
        日誌檔案路徑
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
    """固定隨機種子。

    Args:
        seed: 隨機種子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Fixed seed={seed}")


def cuda_preinit(device: torch.device, retries: int = 10, sleep_s: float = 2.0):
    """預初始化 CUDA，處理偶爾的啟動失敗。

    Args:
        device: CUDA 裝置
        retries: 最大重試次數
        sleep_s: 重試間隔秒數
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
        val_cache_path: 驗證集快取路徑
        batch_size: batch 大小
        num_workers: 工作執行緒數

    Returns:
        DataLoader
    """
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
    """建立訓練集 DataLoader（含資料增強）。

    Args:
        train_cache_path: 訓練集快取路徑
        batch_size: batch 大小
        num_workers: 工作執行緒數
        snr_remix_prob: SNR remix 機率
        snr_remix_range: SNR remix 範圍
        random_gain_prob: 隨機增益機率
        random_gain_db: 隨機增益 dB
        random_crop_prob: 隨機裁切機率
        random_crop_min_ratio: 最小裁切比例
        time_stretch_prob: 時間拉伸機率
        time_stretch_range: 時間拉伸範圍

    Returns:
        DataLoader
    """
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
                mr_stft_loss_fn, mel_loss_fn, fm_loss_fn,
                scaler=None) -> Dict:
    """No-VQ + Decoder LoRA + Frozen Disc FM 訓練 epoch。

    Args:
        model: TeacherStudentNoVQDecoderLoRA 模型
        dataloader: 訓練 DataLoader
        optimizer: 優化器
        device: 計算裝置
        epoch: 當前 epoch
        config: 訓練設定字典
        mr_stft_loss_fn: 多解析度 STFT loss
        mel_loss_fn: Mel 頻譜 loss
        fm_loss_fn: Frozen Discriminator FM loss（新增）
        scaler: GradScaler（AMP 用）

    Returns:
        訓練指標字典
    """
    model.train()
    model.teacher.backbone.train()   # decoder LoRA dropout active
    model.teacher.head.eval()        # head frozen
    model.student.eval()             # encoder frozen

    metrics = {
        'total_loss': 0.0,
        'wav_mse': 0.0,
        'stft_sc_loss': 0.0,
        'stft_mag_loss': 0.0,
        'mel_loss': 0.0,
        'fm_loss': 0.0,
        'nan_batches': 0,
    }
    n_batches = 0
    nan_count = 0
    max_nan_per_epoch = 10

    lambda_wav = config['lambda_wav']
    lambda_stft = config['lambda_stft']
    lambda_mel = config['lambda_mel']
    lambda_fm = config['lambda_fm']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [no-VQ decoder LoRA + FM]")

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

            T = min(clean_audio.shape[-1], recon_wav.shape[-1])
            recon_t = recon_wav[..., :T]
            clean_t = clean_audio[..., :T]

            wav_mse = F.mse_loss(recon_t, clean_t)
            sc_loss, mag_loss = mr_stft_loss_fn(recon_t, clean_t)
            stft_loss = sc_loss + mag_loss
            mel_loss = mel_loss_fn(recon_t, clean_t)

            # Frozen disc FM — gradient 只流回 decoder LoRA，不流入 disc
            fm_loss = fm_loss_fn(recon_t, clean_t)

            loss = (
                lambda_wav * wav_mse
                + lambda_stft * stft_loss
                + lambda_mel * mel_loss
                + lambda_fm * fm_loss
            ) / config['grad_accum']

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
        metrics['fm_loss'] += fm_loss.item()
        n_batches += 1

        pbar.set_postfix({
            'total': f"{loss_val:.4f}",
            'wav': f"{wav_mse.item():.5f}",
            'stft': f"{(sc_loss + mag_loss).item():.3f}",
            'mel': f"{mel_loss.item():.3f}",
            'fm': f"{fm_loss.item():.4f}",
        })

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
                   'mel_loss', 'fm_loss']:
            metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, config,
             mr_stft_loss_fn, mel_loss_fn, fm_loss_fn,
             max_batches=30) -> Dict:
    """驗證集評估。

    Args:
        model: 模型
        dataloader: 驗證 DataLoader
        device: 計算裝置
        config: 設定
        mr_stft_loss_fn: MR-STFT loss
        mel_loss_fn: Mel loss
        fm_loss_fn: Frozen Disc FM loss
        max_batches: 最大 batch 數

    Returns:
        驗證指標字典
    """
    model.eval()

    wav_mse_list, noisy_mse_list = [], []
    stft_sc_list, stft_mag_list, mel_list, fm_list = [], [], [], []
    noisy_stft_sc_list, noisy_mel_list = [], []

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
        sc, mag = mr_stft_loss_fn(recon_t, clean_t)
        stft_sc_list.append(sc.item())
        stft_mag_list.append(mag.item())
        mel_list.append(mel_loss_fn(recon_t, clean_t).item())
        fm_list.append(fm_loss_fn(recon_t, clean_t).item())

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
        'val_fm_loss': float(np.mean(fm_list)) if fm_list else float('nan'),
        'val_noisy_stft_sc': float(np.mean(noisy_stft_sc_list)) if noisy_stft_sc_list else float('nan'),
        'val_noisy_mel': float(np.mean(noisy_mel_list)) if noisy_mel_list else float('nan'),
    }


def _save_audio_samples(model, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    """儲存音訊樣本（noisy / recon / clean）。

    Args:
        model: 模型
        loader: DataLoader
        device: 計算裝置
        output_dir: 輸出目錄
        epoch: 當前 epoch
        num_samples: 樣本數量
        split: 'train' 或 'val'
    """
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
    """繪製訓練曲線圖。

    Args:
        history: 訓練歷史字典
        output_dir: 輸出目錄
        epoch: 當前 epoch
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f'exp_0225d: No-VQ + Decoder LoRA + Frozen Disc FM (Epoch {epoch})', fontsize=14)

    epochs = range(1, len(history['train_total_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train total', alpha=0.8)
    ax.set_title('Total Loss (train)')
    ax.set_yscale('log')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    if history.get('val_noisy_mse'):
        ax.plot(epochs, history['val_noisy_mse'], 'gray', ls='--', label='Noisy baseline')
    if history.get('val_wav_mse'):
        ax.plot(epochs, history['val_wav_mse'], 'r-', label='Recon vs Clean')
    ax.set_title('Wav MSE (val)')
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    if history.get('train_stft_sc'):
        ax.plot(epochs, history['train_stft_sc'], 'c-', label='SC loss')
    if history.get('train_stft_mag'):
        ax.plot(epochs, history['train_stft_mag'], 'm-', label='Mag loss')
    ax.set_title('STFT Losses (train)')
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    if history.get('train_mel'):
        ax.plot(epochs, history['train_mel'], 'orange', label='Train mel')
    if history.get('val_mel_loss'):
        ax.plot(epochs, history['val_mel_loss'], 'r-', label='Val mel')
    if history.get('val_noisy_mel'):
        ax.plot(epochs, history['val_noisy_mel'], 'gray', ls='--', label='Noisy mel')
    ax.set_title('Mel Loss')
    ax.legend(); ax.grid(True)

    # FM loss curves (NEW)
    ax = axes[2, 0]
    if history.get('train_fm'):
        ax.plot(epochs, history['train_fm'], 'darkred', label='Train FM loss')
    if history.get('val_fm_loss'):
        ax.plot(epochs, history['val_fm_loss'], 'red', label='Val FM loss')
    ax.set_title('Feature Matching Loss')
    ax.legend(); ax.grid(True)

    ax = axes[2, 1]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'green', linewidth=2)
    ax.set_title('Learning Rate'); ax.grid(True)

    ax = axes[3, 0]
    if history.get('val_wav_mse') and history.get('val_noisy_mse'):
        improvement = [
            (nm - rm) / nm * 100
            for nm, rm in zip(history['val_noisy_mse'], history['val_wav_mse'])
        ]
        ax.plot(epochs[:len(improvement)], improvement, 'purple', linewidth=2)
        ax.axhline(y=0, color='gray', ls='--')
        ax.set_title('MSE Improvement over Noisy (%)')
    ax.grid(True)

    ax = axes[3, 1]
    if history.get('val_fm_loss') and history.get('train_fm'):
        ax.plot(epochs, history['train_fm'], 'darkred', alpha=0.5, label='Train')
        ax.plot(epochs, history['val_fm_loss'], 'red', label='Val')
        ax.set_title('FM Loss (Train vs Val)')
        ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch{epoch:03d}.png', dpi=150)
    plt.close()
    print(f"  Loss plot saved: training_curves_epoch{epoch:03d}.png")


# ============================================================
# Main
# ============================================================

def main():
    """exp_0225d 主函數：No-VQ + Decoder LoRA + Frozen Disc FM 訓練。"""
    parser = argparse.ArgumentParser(
        description='exp_0225d: No-VQ + Decoder LoRA + Frozen Discriminator FM'
    )

    parser.add_argument('--mode', type=str, default='smoke',
                        choices=['smoke', 'epoch'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Encoder LoRA（只用於載入 checkpoint，不訓練）
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    # Decoder LoRA
    parser.add_argument('--decoder_lora_rank', type=int, default=32)
    parser.add_argument('--decoder_lora_alpha', type=int, default=64)
    parser.add_argument('--decoder_lora_dropout', type=float, default=0.1)

    # Loss weights
    parser.add_argument('--lambda_wav', type=float, default=1.0)
    parser.add_argument('--lambda_stft', type=float, default=1.0)
    parser.add_argument('--lambda_mel', type=float, default=45.0)
    parser.add_argument('--lambda_fm', type=float, default=2.0,
                        help='Frozen disc FM loss 權重。原始 WavTokenizer 中 FM 約佔 '
                             '總 loss 的 5-10%%。建議從 2.0 開始調整。')

    # Discriminator selection
    parser.add_argument('--use_mpd', action='store_true', default=True,
                        help='使用 MultiPeriodDiscriminator')
    parser.add_argument('--use_mrd', action='store_true', default=True,
                        help='使用 MultiResolutionDiscriminator')
    parser.add_argument('--use_dac', action='store_true', default=True,
                        help='使用 DACDiscriminator（batch_size=4 + grad_accum=4 已節省足夠 VRAM）')
    parser.add_argument('--no_mpd', action='store_true', default=False)
    parser.add_argument('--no_mrd', action='store_true', default=False)
    parser.add_argument('--no_dac', action='store_true', default=False)

    parser.add_argument('--stft_fft_sizes', type=str, default='2048,1024,512')
    parser.add_argument('--stft_hop_sizes', type=str, default='512,256,128')
    parser.add_argument('--stft_win_sizes', type=str, default='2048,1024,512')

    parser.add_argument('--encoder_ckpt', type=str, default=str(EXP0217_BEST_CKPT),
                        help='Encoder checkpoint 路徑。應指定 exp_0225a best_model_val_total.pt')

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

    # 處理 --no_* flags
    if args.no_mpd:
        args.use_mpd = False
    if args.no_mrd:
        args.use_mrd = False
    if args.no_dac:
        args.use_dac = False

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
        exp_dir = Path(f'exp_0225/runs/no_vq_scratch_dec_lora_fm_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0225d_no_vq_scratch_decoder_lora_fm'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0225d: No-VQ + Decoder LoRA + Frozen Discriminator FM")
    print("=" * 70)
    print(f"Mode: {args.mode} | Epochs: {args.epochs} | Seed: {args.seed}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LR: {args.learning_rate} → {args.min_lr}")
    print(f"Encoder LoRA: rank={args.lora_rank} (FROZEN, from exp_0225a)")
    print(f"VQ: SKIPPED")
    print(f"Decoder LoRA: rank={args.decoder_lora_rank}, alpha={args.decoder_lora_alpha} (TRAINABLE)")
    print(f"Discriminators: MPD={'✓' if args.use_mpd else '✗'}  "
          f"MRD={'✓' if args.use_mrd else '✗'}  "
          f"DAC={'✓' if args.use_dac else '✗'}  (ALL FROZEN)")
    print(f"Loss: λ_wav={args.lambda_wav} + λ_stft={args.lambda_stft} "
          f"+ λ_mel={args.lambda_mel} + λ_fm={args.lambda_fm}")
    print(f"Encoder ckpt: {args.encoder_ckpt}")
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
        smoke_ds = Subset(full_ds, list(range(min(20, len(full_ds)))))
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

    print("\nBuilding TeacherStudentNoVQDecoderLoRA...")
    model = TeacherStudentNoVQDecoderLoRA(
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

    encoder_ckpt_path = Path(args.encoder_ckpt)
    if encoder_ckpt_path.exists():
        print(f"\nLoading encoder checkpoint: {encoder_ckpt_path}")
        model.load_encoder_checkpoint(str(encoder_ckpt_path))
    else:
        print(f"\n[WARNING] Encoder checkpoint not found: {encoder_ckpt_path}")

    mr_stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_sizes=win_sizes,
    ).to(device)
    mel_loss_fn = MelReconstructionLoss(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100,
    ).to(device)

    print("\nBuilding FrozenDiscriminatorFMLoss...")
    fm_loss_fn = FrozenDiscriminatorFMLoss(
        wavtok_ckpt_path=WAVTOK_FULL_CKPT,
        use_mpd=args.use_mpd,
        use_mrd=args.use_mrd,
        use_dac=args.use_dac,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)

    disc_params = sum(p.numel() for p in fm_loss_fn.parameters())
    print(f"\nTrainable params: {trainable_count:,} / {total_params:,} "
          f"({100*trainable_count/total_params:.3f}%)")
    print(f"Frozen disc params: {disc_params:,}")
    assert trainable_count > 0, "No trainable params found!"

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        """Cosine annealing with warmup 學習率排程。

        Args:
            epoch: 當前 epoch

        Returns:
            學習率倍率
        """
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.learning_rate,
                   0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if args.use_amp else None

    best_val_mse = float('inf')
    history = {
        'train_total_loss': [], 'train_wav_mse': [],
        'train_stft_sc': [], 'train_stft_mag': [], 'train_mel': [],
        'train_fm': [],
        'val_wav_mse': [], 'val_noisy_mse': [],
        'val_stft_sc': [], 'val_mel_loss': [],
        'val_fm_loss': [],
        'val_noisy_stft_sc': [], 'val_noisy_mel': [],
        'lr': [],
    }

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config,
            mr_stft_loss_fn, mel_loss_fn, fm_loss_fn, scaler,
        )
        val_metrics = evaluate(
            model, val_loader, device, config,
            mr_stft_loss_fn, mel_loss_fn, fm_loss_fn, args.eval_max_batches,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_wav_mse'].append(train_metrics['wav_mse'])
        history['train_stft_sc'].append(train_metrics['stft_sc_loss'])
        history['train_stft_mag'].append(train_metrics['stft_mag_loss'])
        history['train_mel'].append(train_metrics['mel_loss'])
        history['train_fm'].append(train_metrics['fm_loss'])
        history['val_wav_mse'].append(val_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(val_metrics['val_noisy_mse'])
        history['val_stft_sc'].append(val_metrics['val_stft_sc'])
        history['val_mel_loss'].append(val_metrics['val_mel_loss'])
        history['val_fm_loss'].append(val_metrics['val_fm_loss'])
        history['val_noisy_stft_sc'].append(val_metrics['val_noisy_stft_sc'])
        history['val_noisy_mel'].append(val_metrics['val_noisy_mel'])
        history['lr'].append(current_lr)

        val_mse = val_metrics['val_wav_mse']
        noisy_mse = val_metrics['val_noisy_mse']
        improve_pct = (noisy_mse - val_mse) / noisy_mse * 100 if noisy_mse > 0 else 0

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: total={train_metrics['total_loss']:.4f}  "
              f"wav={train_metrics['wav_mse']:.5f}  "
              f"stft={train_metrics['stft_sc_loss']+train_metrics['stft_mag_loss']:.3f}  "
              f"mel={train_metrics['mel_loss']:.3f}  "
              f"fm={train_metrics['fm_loss']:.4f}")
        print(f"  Val:   recon_mse={val_mse:.5f}  noisy_mse={noisy_mse:.5f}  "
              f"mse_improve=+{improve_pct:.1f}%")
        print(f"         recon_mel={val_metrics['val_mel_loss']:.4f}  "
              f"noisy_mel={val_metrics['val_noisy_mel']:.4f}  "
              f"stft_sc={val_metrics['val_stft_sc']:.4f}  "
              f"fm={val_metrics['val_fm_loss']:.4f}")
        print(f"  LR={current_lr:.3e}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_ckpt = {
                'epoch': epoch,
                'decoder_lora_state': model.get_decoder_lora_state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'val_wav_mse': val_mse,
                    'val_noisy_mse': noisy_mse,
                    'improve_pct': improve_pct,
                    'val_fm_loss': val_metrics['val_fm_loss'],
                },
                'config': config,
            }
            torch.save(best_ckpt, exp_dir / 'best_model.pt')
            print(f"  ★ New best val MSE: {best_val_mse:.5f} → saved best_model.pt")

        if epoch % args.save_checkpoint_every == 0:
            ckpt = {
                'epoch': epoch,
                'decoder_lora_state': model.get_decoder_lora_state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }
            torch.save(ckpt, exp_dir / f'checkpoint_epoch{epoch:03d}.pt')

        if epoch % args.save_audio_interval == 0 or epoch == args.epochs:
            plot_training_curves(history, exp_dir, epoch)
            _save_audio_samples(model, val_loader, device, exp_dir, epoch,
                                num_samples=2, split='val')
            _save_audio_samples(model, train_loader, device, exp_dir, epoch,
                                num_samples=2, split='train')

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val MSE: {best_val_mse:.5f}")
    print(f"Output: {exp_dir}")


if __name__ == '__main__':
    main()

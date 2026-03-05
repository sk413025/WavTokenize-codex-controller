"""
exp_0228: No-VQ Encoder LoRA + FeatAlign + Frozen MRD FM + Frozen HuBERT
          + Latent Distribution Alignment (CORAL / MMD)

在 exp_0228 基礎上新增 distribution alignment，目標是降低：
    student encoder 輸出 vs frozen decoder 期望輸入 的統計分佈落差。

新增 Loss：
    + λ_dist * DistAlign(student_encoder_out, teacher_encoder_out)
    DistAlign 支援：
        - CORAL（對齊 covariance）
        - MMD-RBF（對齊分佈）
        - both（CORAL + MMD）

執行：
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python exp_0228/train_enc_hubert_fm_distalign.py --mode smoke

    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python exp_0228/train_enc_hubert_fm_distalign.py \\
        --mode epoch --epochs 300 --device cuda:1 --align_type coral --lambda_dist 0.2
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
from exp_0224.models_no_vq import TeacherStudentNoVQ
from exp_0216.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum
from decoder.discriminators import MultiResolutionDiscriminator
from transformers import HubertModel


# ============================================================
# Constants
# ============================================================

SAMPLE_RATE = 24000
HUBERT_SAMPLE_RATE = 16000

EXP0225A_BEST_CKPT = (
    'exp_0225/runs/no_vq_scratch_epoch_20260224_032104/best_model_val_total.pt'
)


# ============================================================
# Frozen MRD Feature Matching Loss (from exp_0227)
# ============================================================

class FrozenMRDFeatureMatchingLoss(nn.Module):
    """使用 WavTokenizer 官方預訓練的 MultiResolutionDiscriminator，
    計算 recon_wav 與 clean_wav 在感知特徵空間的 L1 距離。

    Discriminator 完全凍結（requires_grad=False），
    梯度仍然流過 frozen MRD 的計算圖，最終到達 encoder LoRA。

    MRD forward: (y_real, y_fake) → (scores_r, scores_g, fmap_rs, fmap_gs)
    FM loss = mean( L1(fmap_g[i][j], fmap_r[i][j]) for all i, j )
    """

    def __init__(self, wavtok_ckpt_path: str, device: torch.device):
        """初始化凍結的 MRD Feature Matching Loss。

        Args:
            wavtok_ckpt_path: WavTokenizer checkpoint 路徑，用於載入 MRD 權重。
            device: 計算裝置。
        """
        super().__init__()
        self.mrd = MultiResolutionDiscriminator()

        ckpt = torch.load(wavtok_ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        mrd_state = {}
        for k, v in state.items():
            if 'mrd.' in k:
                new_key = k.split('mrd.', 1)[1]
                mrd_state[new_key] = v
        if mrd_state:
            missing, unexpected = self.mrd.load_state_dict(mrd_state, strict=False)
            print(f"[MRD] Loaded {len(mrd_state)} keys "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("[MRD] WARNING: no MRD keys found in checkpoint")

        for p in self.mrd.parameters():
            p.requires_grad = False
        self.mrd.eval()

        total_params = sum(p.numel() for p in self.mrd.parameters())
        print(f"[MRD] Frozen params: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, recon_wav: torch.Tensor, clean_wav: torch.Tensor) -> torch.Tensor:
        """計算 MRD feature matching loss。

        Args:
            recon_wav: 重建音訊波形 [B, 1, T] 或 [B, T]。
            clean_wav: 乾淨參考波形 [B, 1, T] 或 [B, T]。

        Returns:
            feature matching loss（L1 距離的均值）。
        """
        if recon_wav.dim() == 3:
            recon_wav = recon_wav.squeeze(1)
        if clean_wav.dim() == 3:
            clean_wav = clean_wav.squeeze(1)

        # MRD forward: (y_real, y_fake) → (scores_r, scores_g, fmap_rs, fmap_gs)
        # fmap_rs = clean feature maps, fmap_gs = recon feature maps
        # MRD 完全凍結，但 recon_wav 的梯度會流過計算圖到達 encoder LoRA
        # DiscriminatorR.spectrogram 內部使用 torch.stft → 需要 2D [B, T] 輸入
        _, _, fmap_rs, fmap_gs = self.mrd(
            clean_wav.detach(),
            recon_wav,
        )

        fm_loss = 0.0
        n_maps = 0
        for fmap_r_disc, fmap_g_disc in zip(fmap_rs, fmap_gs):
            for fr, fg in zip(fmap_r_disc, fmap_g_disc):
                fm_loss += F.l1_loss(fg, fr.detach())
                n_maps += 1
        if n_maps > 0:
            fm_loss = fm_loss / n_maps
        return fm_loss


# ============================================================
# Frozen HuBERT Feature Loss (NEW)
# ============================================================

class FrozenHuBERTFeatureLoss(nn.Module):
    """使用凍結的 HuBERT-base-ls960 計算語音學特徵距離。

    從 HuBERT 中間層（layer 6-8）提取語音學特徵，
    計算重建音訊與乾淨音訊在語音學空間的 L1 距離。

    HuBERT 中間層編碼了豐富的語音學資訊（phonemes, manner, place of articulation），
    能有效監督子音清晰度與發音保真度。

    輸入 24kHz → resample 至 16kHz → HuBERT → 取中間層特徵均值 → L1
    """

    def __init__(self, model_name: str = 'facebook/hubert-base-ls960',
                 layers: Tuple[int, ...] = (6, 7, 8), device: torch.device = None):
        """初始化凍結的 HuBERT Feature Loss。

        Args:
            model_name: HuggingFace 模型名稱。
            layers: 要提取的 HuBERT 隱藏層索引（0-based, 0=embedding, 1-12=transformer layers）。
            device: 計算裝置。
        """
        super().__init__()
        self.layers = layers

        print(f"[HuBERT] Loading {model_name}...")
        self.hubert = HubertModel.from_pretrained(model_name, use_safetensors=True)

        for p in self.hubert.parameters():
            p.requires_grad = False
        self.hubert.eval()

        total_params = sum(p.numel() for p in self.hubert.parameters())
        print(f"[HuBERT] Frozen params: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"[HuBERT] Using layers: {layers} for phonetic features")
        print(f"[HuBERT] Input: 16kHz mono → 50Hz frame-level features (dim=768)")

    def _resample_to_16k(self, wav: torch.Tensor) -> torch.Tensor:
        """將 24kHz 波形降採樣至 16kHz。

        Args:
            wav: 輸入波形 [B, T_24k]（24kHz）。

        Returns:
            降採樣後的波形 [B, T_16k]（16kHz）。
        """
        return torchaudio.functional.resample(
            wav, orig_freq=SAMPLE_RATE, new_freq=HUBERT_SAMPLE_RATE
        )

    def _extract_features(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """從 HuBERT 中間層提取語音學特徵。

        Args:
            wav_16k: 16kHz 波形 [B, T_16k]。

        Returns:
            語音學特徵 [B, T_frames, 768]，為指定層的均值。
        """
        outputs = self.hubert(wav_16k, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, T, 768), length=13

        selected = [hidden_states[i] for i in self.layers]
        return torch.stack(selected, dim=0).mean(dim=0)  # [B, T_frames, 768]

    def forward(self, recon_wav: torch.Tensor, clean_wav: torch.Tensor) -> torch.Tensor:
        """計算 HuBERT feature loss。

        Args:
            recon_wav: 重建音訊波形 [B, 1, T] 或 [B, T]（24kHz）。
            clean_wav: 乾淨參考波形 [B, 1, T] 或 [B, T]（24kHz）。

        Returns:
            HuBERT 語音學特徵的 L1 距離。
        """
        if recon_wav.dim() == 3:
            recon_wav = recon_wav.squeeze(1)
        if clean_wav.dim() == 3:
            clean_wav = clean_wav.squeeze(1)

        # resample 24kHz → 16kHz
        recon_16k = self._resample_to_16k(recon_wav)
        clean_16k = self._resample_to_16k(clean_wav)

        # 確保長度一致
        T = min(recon_16k.shape[-1], clean_16k.shape[-1])
        recon_16k = recon_16k[..., :T]
        clean_16k = clean_16k[..., :T]

        # clean side: no grad needed (frozen teacher)
        with torch.no_grad():
            clean_feat = self._extract_features(clean_16k)

        # recon side: grad flows through resample + HuBERT to encoder LoRA
        recon_feat = self._extract_features(recon_16k)

        # align frame lengths
        Tf = min(recon_feat.shape[1], clean_feat.shape[1])
        return F.l1_loss(recon_feat[:, :Tf], clean_feat[:, :Tf].detach())


# ============================================================
# Loss functions (from exp_0227)
# ============================================================

class STFTLoss(nn.Module):
    """單一解析度 STFT loss（Spectral Convergence + Log Magnitude）。"""

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        """初始化 STFT Loss。

        Args:
            n_fft: FFT 大小。
            hop_length: hop 步幅。
            win_length: 窗口大小。
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算 STFT spectral convergence 和 magnitude loss。

        Args:
            y_hat: 預測波形 [B, T]。
            y: 目標波形 [B, T]。

        Returns:
            (spectral_convergence_loss, magnitude_loss) 元組。
        """
        if self.window.device != y_hat.device:
            self.window = self.window.to(y_hat.device)
        stft_hat = torch.stft(y_hat, self.n_fft, self.hop_length, self.win_length,
                              self.window, return_complex=True)
        stft_y   = torch.stft(y, self.n_fft, self.hop_length, self.win_length,
                              self.window, return_complex=True)
        mag_hat = stft_hat.abs().clamp(min=1e-7)
        mag_y   = stft_y.abs().clamp(min=1e-7)
        sc_loss  = torch.norm(mag_y - mag_hat, p='fro') / torch.norm(mag_y, p='fro').clamp(min=1e-7)
        mag_loss = F.l1_loss(torch.log(mag_hat), torch.log(mag_y))
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """多解析度 STFT loss。"""

    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512],
        hop_sizes: List[int] = [512, 256, 128],
        win_sizes: List[int] = [2048, 1024, 512],
    ):
        """初始化多解析度 STFT Loss。

        Args:
            fft_sizes: 各解析度的 FFT 大小列表。
            hop_sizes: 各解析度的 hop 步幅列表。
            win_sizes: 各解析度的窗口大小列表。
        """
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft, hop, win)
            for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算多解析度 STFT loss。

        Args:
            y_hat: 預測波形 [B, T] 或 [B, 1, T]。
            y: 目標波形 [B, T] 或 [B, 1, T]。

        Returns:
            (mean_sc_loss, mean_mag_loss) 元組。
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
    """Mel 頻譜重建 loss。"""

    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        """初始化 Mel 重建 Loss。

        Args:
            sample_rate: 取樣率。
            n_fft: FFT 大小。
            hop_length: hop 步幅。
            n_mels: Mel 濾波器數量。
        """
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, center=True, power=1,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """計算 Mel 頻譜重建 loss。

        Args:
            y_hat: 預測波形 [B, T] 或 [B, 1, T]。
            y: 目標波形 [B, T] 或 [B, 1, T]。

        Returns:
            Mel 頻譜的 L1 距離。
        """
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        mel_hat = torch.log(self.mel_spec(y_hat).clamp(min=1e-7))
        mel = torch.log(self.mel_spec(y).clamp(min=1e-7))
        return F.l1_loss(mel, mel_hat)


# ============================================================
# Latent Distribution Alignment Loss
# ============================================================

def _flatten_latent_for_align(
    feat: torch.Tensor,
    max_tokens: int = 1024,
) -> torch.Tensor:
    """將 [B, C, T] latent 轉成 [N, C] 並可選擇子抽樣 token。"""
    flat = feat.transpose(1, 2).reshape(-1, feat.shape[1])
    if max_tokens > 0 and flat.shape[0] > max_tokens:
        idx = torch.randperm(flat.shape[0], device=flat.device)[:max_tokens]
        flat = flat[idx]
    return flat


def coral_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    max_tokens: int = 1024,
    eps: float = 1e-5,
) -> torch.Tensor:
    """CORAL: 對齊 student/teacher latent 的二階統計（covariance）。"""
    x = _flatten_latent_for_align(student_feat, max_tokens=max_tokens)
    y = _flatten_latent_for_align(teacher_feat, max_tokens=max_tokens)

    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return torch.zeros((), device=student_feat.device, dtype=student_feat.dtype)
    if x.shape[0] != n:
        x = x[:n]
    if y.shape[0] != n:
        y = y[:n]

    # 強制 fp32 避免 fp16 covariance matmul 溢出
    x = x.float()
    y = y.float()

    # 限制特徵值範圍，防止極端 batch 產生 NaN covariance
    x = x.clamp(-100.0, 100.0)
    y = y.clamp(-100.0, 100.0)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    d = x.shape[1]
    eye = torch.eye(d, device=x.device, dtype=torch.float32)
    cov_x = (x.t() @ x) / (n - 1) + eps * eye
    cov_y = (y.t() @ y) / (n - 1) + eps * eye

    # 標準 CORAL 正規化：除以 4d² 使 loss 與 feature dim 無關
    loss = ((cov_x - cov_y) ** 2).sum() / (4.0 * d * d)
    # 最終安全檢查：若仍為 NaN 回傳 0
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.zeros((), device=student_feat.device, dtype=student_feat.dtype)
    return loss


def mmd_rbf_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    sigma: float = 1.0,
    max_tokens: int = 512,
) -> torch.Tensor:
    """RBF-MMD: 對齊 student/teacher latent 分佈。"""
    x = _flatten_latent_for_align(student_feat, max_tokens=max_tokens)
    y = _flatten_latent_for_align(teacher_feat, max_tokens=max_tokens)

    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return torch.zeros((), device=student_feat.device, dtype=student_feat.dtype)
    if x.shape[0] != n:
        x = x[:n]
    if y.shape[0] != n:
        y = y[:n]

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    gamma = 1.0 / (2.0 * sigma * sigma)

    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    k_xx = torch.exp(-gamma * xx)
    k_yy = torch.exp(-gamma * yy)
    k_xy = torch.exp(-gamma * xy)

    n_float = float(n)
    k_xx = (k_xx.sum() - k_xx.diag().sum()) / (n_float * (n_float - 1.0))
    k_yy = (k_yy.sum() - k_yy.diag().sum()) / (n_float * (n_float - 1.0))
    k_xy = k_xy.mean()
    return torch.clamp(k_xx + k_yy - 2.0 * k_xy, min=0.0)


def distribution_align_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    align_type: str = 'coral',
    mmd_sigma: float = 1.0,
    max_tokens: int = 1024,
) -> torch.Tensor:
    """整合式分佈對齊 loss（none/coral/mmd/both）。"""
    if align_type == 'none':
        return torch.zeros((), device=student_feat.device, dtype=student_feat.dtype)
    if align_type == 'coral':
        return coral_loss(student_feat, teacher_feat, max_tokens=max_tokens)
    if align_type == 'mmd':
        return mmd_rbf_loss(
            student_feat, teacher_feat, sigma=mmd_sigma, max_tokens=max_tokens
        )
    if align_type == 'both':
        l_coral = coral_loss(student_feat, teacher_feat, max_tokens=max_tokens)
        l_mmd = mmd_rbf_loss(
            student_feat, teacher_feat, sigma=mmd_sigma,
            max_tokens=max(1, max_tokens // 2),
        )
        return l_coral + l_mmd
    raise ValueError(f"Unsupported align_type: {align_type}")


# ============================================================
# Utilities
# ============================================================

class _TeeIO:
    """同時輸出到多個 stream 的 IO wrapper。"""

    def __init__(self, *streams):
        """初始化 TeeIO。

        Args:
            *streams: 要同時寫入的 stream 列表。
        """
        self._streams = streams

    def write(self, data):
        """寫入資料到所有 stream。

        Args:
            data: 要寫入的字串。
        """
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        """刷新所有 stream 的緩衝區。"""
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        """檢查是否為終端裝置。

        Returns:
            永遠回傳 False。
        """
        return False


def setup_logging(output_dir: Path) -> Path:
    """設定日誌系統，將 stdout/stderr 同時輸出到檔案。

    Args:
        output_dir: 輸出目錄路徑。

    Returns:
        日誌檔案路徑，若設定失敗則回傳 None。
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
    """設定隨機種子以確保可重現性。

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
    """預初始化 CUDA 裝置，帶重試機制。

    Args:
        device: 目標 CUDA 裝置。
        retries: 最大重試次數。
        sleep_s: 重試間隔秒數。
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
        num_workers: 資料載入工作數。

    Returns:
        驗證集的 DataLoader。
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
        train_cache_path: 訓練集快取路徑。
        batch_size: 批次大小。
        num_workers: 資料載入工作數。
        snr_remix_prob: SNR 混音機率。
        snr_remix_range: SNR 混音範圍 (min, max)。
        random_gain_prob: 隨機增益機率。
        random_gain_db: 隨機增益幅度 (dB)。
        random_crop_prob: 隨機裁切機率。
        random_crop_min_ratio: 最小裁切比例。
        time_stretch_prob: 時間拉伸機率。
        time_stretch_range: 時間拉伸範圍 (min, max)。

    Returns:
        訓練集的 DataLoader。
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

def train_epoch(model, mrd_fm_loss_fn, hubert_loss_fn, dataloader, optimizer,
                device, epoch, config, mr_stft_loss_fn, mel_loss_fn,
                scaler=None) -> Dict:
    """執行一個 epoch 的訓練。

    Args:
        model: TeacherStudentNoVQ 模型。
        mrd_fm_loss_fn: 凍結 MRD feature matching loss。
        hubert_loss_fn: 凍結 HuBERT feature loss。
        dataloader: 訓練集 DataLoader。
        optimizer: 優化器。
        device: 計算裝置。
        epoch: 當前 epoch 編號。
        config: 訓練設定字典。
        mr_stft_loss_fn: 多解析度 STFT loss。
        mel_loss_fn: Mel 重建 loss。
        scaler: AMP GradScaler（可選）。

    Returns:
        包含各項 loss 平均值的字典。
    """
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
        'dist_align_loss': 0.0,
        'mrd_fm_loss': 0.0,
        'hubert_loss': 0.0,
        'nan_batches': 0,
    }
    n_batches = 0
    nan_count = 0
    max_nan_per_epoch = 10

    lambda_wav    = config['lambda_wav']
    lambda_stft   = config['lambda_stft']
    lambda_mel    = config['lambda_mel']
    lambda_feat   = config['lambda_feat']
    lambda_dist   = config['lambda_dist']
    lambda_fm     = config['lambda_fm']
    lambda_hubert = config['lambda_hubert']
    align_type    = config['align_type']
    mmd_sigma     = config['mmd_sigma']
    dist_max_tokens = config['dist_max_tokens']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [enc+mrd-fm+hubert+dist]")

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
            recon_wav    = out['recon_wav']
            student_feat = out['student_encoder_out']
            teacher_feat = out['teacher_encoder_out']

            T = min(clean_audio.shape[-1], recon_wav.shape[-1])
            recon_t = recon_wav[..., :T]
            clean_t = clean_audio[..., :T]

            # waveform losses
            wav_mse = F.mse_loss(recon_t, clean_t)
            sc_loss, mag_loss = mr_stft_loss_fn(recon_t, clean_t)
            stft_loss = sc_loss + mag_loss
            mel_loss = mel_loss_fn(recon_t, clean_t)

            # encoder feature alignment
            Tf = min(student_feat.shape[-1], teacher_feat.shape[-1])
            student_trim = student_feat[..., :Tf]
            teacher_trim = teacher_feat[..., :Tf].detach()
            feat_align = F.mse_loss(
                student_trim,
                teacher_trim,
            )
            dist_align = distribution_align_loss(
                student_trim,
                teacher_trim,
                align_type=align_type,
                mmd_sigma=mmd_sigma,
                max_tokens=dist_max_tokens,
            )

            # frozen MRD feature matching（感知音質梯度）
            mrd_fm = mrd_fm_loss_fn(recon_t, clean_t)

            # frozen HuBERT feature loss（語音學內容梯度）
            hubert_loss = hubert_loss_fn(recon_t, clean_t)

            loss = (
                lambda_wav    * wav_mse
                + lambda_stft * stft_loss
                + lambda_mel  * mel_loss
                + lambda_feat * feat_align
                + lambda_dist * dist_align
                + lambda_fm   * mrd_fm
                + lambda_hubert * hubert_loss
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
        metrics['total_loss']      += loss_val
        metrics['wav_mse']         += wav_mse.item()
        metrics['stft_sc_loss']    += sc_loss.item()
        metrics['stft_mag_loss']   += mag_loss.item()
        metrics['mel_loss']        += mel_loss.item()
        metrics['feat_align_loss'] += feat_align.item()
        metrics['dist_align_loss'] += dist_align.item()
        metrics['mrd_fm_loss']     += mrd_fm.item()
        metrics['hubert_loss']     += hubert_loss.item()
        n_batches += 1

        pbar.set_postfix({
            'total':  f"{loss_val:.4f}",
            'wav':    f"{wav_mse.item():.5f}",
            'dist':   f"{dist_align.item():.4f}",
            'mrd_fm': f"{mrd_fm.item():.4f}",
            'hubert': f"{hubert_loss.item():.4f}",
        })

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
                  'mel_loss', 'feat_align_loss', 'dist_align_loss',
                  'mrd_fm_loss', 'hubert_loss']:
            metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def evaluate(model, mrd_fm_loss_fn, hubert_loss_fn, dataloader, device, config,
             mr_stft_loss_fn, mel_loss_fn, max_batches=30) -> Dict:
    """執行驗證集評估。

    Args:
        model: TeacherStudentNoVQ 模型。
        mrd_fm_loss_fn: 凍結 MRD feature matching loss。
        hubert_loss_fn: 凍結 HuBERT feature loss。
        dataloader: 驗證集 DataLoader。
        device: 計算裝置。
        config: 訓練設定字典。
        mr_stft_loss_fn: 多解析度 STFT loss。
        mel_loss_fn: Mel 重建 loss。
        max_batches: 最大評估 batch 數。

    Returns:
        包含各項驗證指標的字典。
    """
    model.eval()

    wav_mse_list, noisy_mse_list = [], []
    stft_sc_list, stft_mag_list, mel_list = [], [], []
    noisy_stft_sc_list, noisy_mel_list = [], []
    feat_align_list, dist_align_list, mrd_fm_list, hubert_list = [], [], [], []

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
        recon_wav    = out['recon_wav']
        student_feat = out['student_encoder_out']
        teacher_feat = out['teacher_encoder_out']

        T = min(clean_audio.shape[-1], recon_wav.shape[-1], noisy_audio.shape[-1])
        clean_t  = clean_audio[..., :T]
        recon_t  = recon_wav[..., :T]
        noisy_t  = noisy_audio[..., :T]

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
        student_trim = student_feat[..., :Tf]
        teacher_trim = teacher_feat[..., :Tf]
        feat_align_list.append(
            F.mse_loss(student_trim, teacher_trim).item()
        )
        dist_align_list.append(
            distribution_align_loss(
                student_trim,
                teacher_trim,
                align_type=config['align_type'],
                mmd_sigma=config['mmd_sigma'],
                max_tokens=config['dist_max_tokens'],
            ).item()
        )
        mrd_fm_list.append(mrd_fm_loss_fn(recon_t, clean_t).item())
        hubert_list.append(hubert_loss_fn(recon_t, clean_t).item())

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
        'val_dist_align':    float(np.mean(dist_align_list))    if dist_align_list else float('nan'),
        'val_mrd_fm':        float(np.mean(mrd_fm_list))        if mrd_fm_list else float('nan'),
        'val_hubert':        float(np.mean(hubert_list))        if hubert_list else float('nan'),
    }


def _save_audio_samples(model, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    """儲存音訊範例用於人工檢聽。

    Args:
        model: TeacherStudentNoVQ 模型。
        loader: 資料 DataLoader。
        device: 計算裝置。
        output_dir: 輸出目錄。
        epoch: 當前 epoch 編號。
        num_samples: 要儲存的樣本數。
        split: 資料集分割名稱（'val' 或 'train'）。
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
    """繪製訓練曲線圖並儲存。

    Args:
        history: 訓練歷史字典。
        output_dir: 輸出目錄。
        epoch: 當前 epoch 編號。
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(
        f'exp_0228: EncOnly + FeatAlign + DistAlign + MRD FM + HuBERT (Epoch {epoch})',
        fontsize=14,
    )
    epochs = range(1, len(history['train_total_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train total', alpha=0.8)
    if history.get('val_total'):
        ax.plot(epochs[:len(history['val_total'])], history['val_total'], 'r--', label='Val total', alpha=0.8)
    ax.set_title('Total Loss (train & val)')
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
    if history.get('train_mrd_fm'):
        ax.plot(epochs, history['train_mrd_fm'], 'orange', label='Train MRD FM', alpha=0.8)
    if history.get('val_mrd_fm'):
        ax.plot(epochs, history['val_mrd_fm'], 'red', ls='--', label='Val MRD FM', alpha=0.8)
    ax.set_title('MRD Feature Matching Loss (感知音質梯度)')
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    if history.get('train_feat_align'):
        ax.plot(epochs, history['train_feat_align'], 'b-', label='Train feat align')
    if history.get('val_feat_align'):
        ax.plot(epochs, history['val_feat_align'], 'r--', label='Val feat align')
    if history.get('train_dist_align'):
        ax.plot(epochs, history['train_dist_align'], 'c-', label='Train dist align')
    if history.get('val_dist_align'):
        ax.plot(epochs, history['val_dist_align'], 'm--', label='Val dist align')
    ax.set_title('Encoder Alignment Losses (MSE + DistAlign)')
    ax.legend(); ax.grid(True)

    # NEW: HuBERT loss plot
    ax = axes[2, 0]
    if history.get('train_hubert'):
        ax.plot(epochs, history['train_hubert'], 'purple', label='Train HuBERT', alpha=0.8)
    if history.get('val_hubert'):
        ax.plot(epochs, history['val_hubert'], 'magenta', ls='--', label='Val HuBERT', alpha=0.8)
    ax.set_title('HuBERT Feature Loss (語音學內容梯度)')
    ax.legend(); ax.grid(True)

    ax = axes[2, 1]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'green', linewidth=2)
    ax.set_title('Learning Rate')
    ax.grid(True)

    ax = axes[3, 0]
    if (history.get('val_wav_mse') and history.get('val_stft_sc')
            and history.get('val_stft_mag') and history.get('val_mel_loss')):
        val_totals = [
            m + sc + mag + 45.0 * mel
            for m, sc, mag, mel in zip(
                history['val_wav_mse'], history['val_stft_sc'],
                history['val_stft_mag'], history['val_mel_loss'],
            )
        ]
        ax.plot(epochs[:len(val_totals)], val_totals, 'purple', linewidth=2, label='val_total')
        best_ep = val_totals.index(min(val_totals))
        ax.axvline(x=best_ep + 1, color='red', ls='--', alpha=0.7,
                   label=f'best ep{best_ep+1}={min(val_totals):.2f}')
        ax.set_title('Val Total Loss')
        ax.legend(fontsize=8)
    ax.grid(True)

    # Summary text
    ax = axes[3, 1]
    ax.axis('off')
    summary_text = (
        f"exp_0228: Encoder LoRA + FeatAlign + DistAlign + MRD FM + HuBERT\n"
        f"Epoch: {epoch}\n"
        f"Train total: {history['train_total_loss'][-1]:.4f}\n"
    )
    if history.get('train_dist_align'):
        summary_text += f"Train DistAlign: {history['train_dist_align'][-1]:.4f}\n"
    if history.get('train_hubert'):
        summary_text += f"Train HuBERT: {history['train_hubert'][-1]:.4f}\n"
    if history.get('train_mrd_fm'):
        summary_text += f"Train MRD FM: {history['train_mrd_fm'][-1]:.4f}\n"
    if history.get('val_wav_mse'):
        summary_text += f"Val MSE: {history['val_wav_mse'][-1]:.5f}\n"
    if history.get('val_dist_align'):
        summary_text += f"Val DistAlign: {history['val_dist_align'][-1]:.4f}\n"
    if history.get('val_hubert'):
        summary_text += f"Val HuBERT: {history['val_hubert'][-1]:.4f}\n"
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch{epoch:03d}.png', dpi=150)
    plt.close()
    print(f"  Loss plot saved: training_curves_epoch{epoch:03d}.png")


# ============================================================
# Main
# ============================================================

def main():
    """主訓練函式：解析參數、建立模型與損失函式、執行訓練迴圈。"""
    parser = argparse.ArgumentParser(
        description='exp_0228: Encoder LoRA + FeatAlign + DistAlign + MRD FM + HuBERT Feature Loss'
    )
    parser.add_argument('--mode', type=str, default='smoke', choices=['smoke', 'epoch'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--encoder_ckpt', type=str, default=EXP0225A_BEST_CKPT)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:1')

    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    parser.add_argument('--lambda_wav',    type=float, default=1.0)
    parser.add_argument('--lambda_stft',   type=float, default=1.0)
    parser.add_argument('--lambda_mel',    type=float, default=45.0)
    parser.add_argument('--lambda_feat',   type=float, default=1.0)
    parser.add_argument('--lambda_dist',   type=float, default=0.2,
                        help='Latent distribution alignment loss weight')
    parser.add_argument('--lambda_fm',     type=float, default=2.0,
                        help='MRD Feature Matching weight')
    parser.add_argument('--lambda_hubert', type=float, default=1.0,
                        help='HuBERT phonetic feature loss weight（語音學內容監督）')

    parser.add_argument('--hubert_layers', type=str, default='6,7,8',
                        help='HuBERT hidden layer indices to use (comma-separated)')
    parser.add_argument('--align_type', type=str, default='coral',
                        choices=['none', 'coral', 'mmd', 'both'],
                        help='Distribution alignment type')
    parser.add_argument('--mmd_sigma', type=float, default=1.0,
                        help='RBF sigma for MMD alignment')
    parser.add_argument('--dist_max_tokens', type=int, default=1024,
                        help='Max latent tokens sampled for DistAlign loss')

    parser.add_argument('--stft_fft_sizes', type=str, default='2048,1024,512')
    parser.add_argument('--stft_hop_sizes', type=str, default='512,256,128')
    parser.add_argument('--stft_win_sizes', type=str, default='2048,1024,512')

    parser.add_argument('--snr_remix_prob', type=float, default=0.5)
    parser.add_argument('--snr_remix_min',  type=float, default=-5.0)
    parser.add_argument('--snr_remix_max',  type=float, default=25.0)
    parser.add_argument('--random_gain_prob', type=float, default=0.3)
    parser.add_argument('--random_gain_db',   type=float, default=3.0)
    parser.add_argument('--random_crop_prob', type=float, default=0.3)
    parser.add_argument('--random_crop_min_ratio', type=float, default=0.7)
    parser.add_argument('--time_stretch_prob', type=float, default=0.2)
    parser.add_argument('--time_stretch_min',  type=float, default=0.95)
    parser.add_argument('--time_stretch_max',  type=float, default=1.05)

    parser.add_argument('--save_checkpoint_every', type=int, default=10)
    parser.add_argument('--save_audio_interval',   type=int, default=25)
    parser.add_argument('--eval_max_batches',      type=int, default=30)

    args = parser.parse_args()

    if args.mode == 'smoke':
        args.epochs = max(args.epochs, 5)
        args.eval_max_batches = 5

    fft_sizes = [int(x) for x in args.stft_fft_sizes.split(',')]
    hop_sizes = [int(x) for x in args.stft_hop_sizes.split(',')]
    win_sizes = [int(x) for x in args.stft_win_sizes.split(',')]
    hubert_layers = tuple(int(x) for x in args.hubert_layers.split(','))

    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(f'exp_0228/runs/enc_hubert_fm_dist_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0228_enc_hubert_fm_distalign'
    config['encoder_init'] = f'exp_0225a best_model_val_total.pt: {args.encoder_ckpt}'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0228: Encoder LoRA + FeatAlign + DistAlign + MRD FM + HuBERT Feature Loss")
    print("=" * 70)
    print(f"Mode: {args.mode} | Epochs: {args.epochs} | Device: {args.device}")
    print(f"LR: {args.learning_rate} → {args.min_lr}")
    print(f"Encoder LoRA: rank={args.lora_rank}, alpha={args.lora_alpha} (trainable)")
    print(f"VQ: SKIPPED | Decoder: FROZEN | MRD: FROZEN | HuBERT: FROZEN")
    print(f"Encoder init: {args.encoder_ckpt}")
    print(f"Loss: λ_wav={args.lambda_wav} + λ_stft={args.lambda_stft} "
          f"+ λ_mel={args.lambda_mel} + λ_feat={args.lambda_feat} "
          f"+ λ_dist={args.lambda_dist} ({args.align_type}) "
          f"+ λ_fm={args.lambda_fm} (MRD FM) "
          f"+ λ_hubert={args.lambda_hubert} (HuBERT layers {hubert_layers})")
    if args.align_type in ['mmd', 'both']:
        print(f"DistAlign details: mmd_sigma={args.mmd_sigma}, dist_max_tokens={args.dist_max_tokens}")
    else:
        print(f"DistAlign details: dist_max_tokens={args.dist_max_tokens}")
    print(f"Output: {exp_dir}")
    if log_path:
        print(f"Log: {log_path}")
    print("=" * 70)

    device = torch.device(args.device)
    cuda_preinit(device)

    print("\nLoading data...")
    if args.mode == 'smoke':
        full_ds = AugmentedCurriculumDataset(
            VAL_CACHE, augment=False, filter_clean_to_clean=True, compute_snr=False,
        )
        smoke_ds = torch.utils.data.Subset(full_ds, list(range(min(20, len(full_ds)))))
        train_loader = DataLoader(smoke_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn_curriculum)
        val_loader   = DataLoader(smoke_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, collate_fn=collate_fn_curriculum)
        print(f"Smoke test: {len(smoke_ds)} samples")
    else:
        train_loader = make_train_loader(
            TRAIN_CACHE, batch_size=args.batch_size, num_workers=2,
            snr_remix_prob=args.snr_remix_prob,
            snr_remix_range=(args.snr_remix_min, args.snr_remix_max),
            random_gain_prob=args.random_gain_prob, random_gain_db=args.random_gain_db,
            random_crop_prob=args.random_crop_prob,
            random_crop_min_ratio=args.random_crop_min_ratio,
            time_stretch_prob=args.time_stretch_prob,
            time_stretch_range=(args.time_stretch_min, args.time_stretch_max),
        )
        val_loader = make_val_loader(VAL_CACHE, batch_size=4, num_workers=2)
        print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    print("\nBuilding TeacherStudentNoVQ (encoder-only trainable)...")
    model = TeacherStudentNoVQ(
        wavtok_config=WAVTOK_CONFIG, wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6], device=device,
    ).to(device)

    ckpt_path = Path(args.encoder_ckpt)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"  Loaded model_state_dict (ep{ckpt.get('epoch','?')})")
            if missing:    print(f"  Missing: {len(missing)}")
            if unexpected: print(f"  Unexpected: {len(unexpected)}")
        elif 'encoder_lora_state' in ckpt:
            model.student.load_state_dict(ckpt['encoder_lora_state'], strict=False)
            print(f"  Loaded encoder_lora_state (ep{ckpt.get('epoch','?')})")
    else:
        print(f"  WARNING: encoder ckpt not found, starting from pretrained WavTokenizer")

    print("\nLoading Frozen MRD discriminator from WavTokenizer ckpt...")
    mrd_fm_loss_fn = FrozenMRDFeatureMatchingLoss(
        wavtok_ckpt_path=str(WAVTOK_CKPT), device=device,
    ).to(device)

    print("\nLoading Frozen HuBERT for phonetic supervision...")
    hubert_loss_fn = FrozenHuBERTFeatureLoss(
        model_name='facebook/hubert-base-ls960',
        layers=hubert_layers, device=device,
    ).to(device)

    mr_stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_sizes=win_sizes,
    ).to(device)
    mel_loss_fn = MelReconstructionLoss(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_hubert = sum(p.numel() for p in hubert_loss_fn.parameters())
    frozen_mrd = sum(p.numel() for p in mrd_fm_loss_fn.parameters())
    print(f"\nTrainable params: {trainable_count:,} / {total_params:,} "
          f"({100*trainable_count/total_params:.3f}%)")
    print(f"Frozen supervisors: HuBERT {frozen_hubert/1e6:.1f}M + MRD {frozen_mrd/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        """計算學習率衰減係數。

        Args:
            epoch: 當前 epoch。

        Returns:
            學習率乘數。
        """
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.learning_rate, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if args.use_amp else None

    best_val_total = float('inf')
    best_val_mse   = float('inf')
    history = {
        'train_total_loss': [], 'train_wav_mse': [],
        'train_stft_sc': [], 'train_stft_mag': [], 'train_mel': [],
        'train_feat_align': [], 'train_dist_align': [],
        'train_mrd_fm': [], 'train_hubert': [],
        'val_wav_mse': [], 'val_noisy_mse': [], 'val_total': [],
        'val_stft_sc': [], 'val_stft_mag': [], 'val_mel_loss': [],
        'val_noisy_stft_sc': [], 'val_noisy_mel': [],
        'val_feat_align': [], 'val_dist_align': [],
        'val_mrd_fm': [], 'val_hubert': [],
        'lr': [],
    }

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, mrd_fm_loss_fn, hubert_loss_fn, train_loader, optimizer,
            device, epoch, config, mr_stft_loss_fn, mel_loss_fn, scaler,
        )
        val_metrics = evaluate(
            model, mrd_fm_loss_fn, hubert_loss_fn, val_loader, device, config,
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
        history['train_dist_align'].append(train_metrics['dist_align_loss'])
        history['train_mrd_fm'].append(train_metrics['mrd_fm_loss'])
        history['train_hubert'].append(train_metrics['hubert_loss'])
        history['val_wav_mse'].append(val_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(val_metrics['val_noisy_mse'])
        history['val_stft_sc'].append(val_metrics['val_stft_sc'])
        history['val_stft_mag'].append(val_metrics['val_stft_mag'])
        history['val_mel_loss'].append(val_metrics['val_mel_loss'])
        history['val_noisy_stft_sc'].append(val_metrics['val_noisy_stft_sc'])
        history['val_noisy_mel'].append(val_metrics['val_noisy_mel'])
        history['val_feat_align'].append(val_metrics['val_feat_align'])
        history['val_dist_align'].append(val_metrics['val_dist_align'])
        history['val_mrd_fm'].append(val_metrics['val_mrd_fm'])
        history['val_hubert'].append(val_metrics['val_hubert'])
        history['lr'].append(current_lr)

        val_mse   = val_metrics['val_wav_mse']
        noisy_mse = val_metrics['val_noisy_mse']
        val_total = (val_mse
                     + val_metrics['val_stft_sc']
                     + val_metrics['val_stft_mag']
                     + args.lambda_mel * val_metrics['val_mel_loss'])
        history['val_total'].append(val_total)
        improve_pct = (noisy_mse - val_mse) / noisy_mse * 100 if noisy_mse > 0 else 0

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: total={train_metrics['total_loss']:.4f}  "
              f"wav={train_metrics['wav_mse']:.5f}  "
              f"feat={train_metrics['feat_align_loss']:.5f}  "
              f"dist={train_metrics['dist_align_loss']:.4f}  "
              f"mrd_fm={train_metrics['mrd_fm_loss']:.4f}  "
              f"hubert={train_metrics['hubert_loss']:.4f}")
        print(f"  Val:   recon_mse={val_mse:.5f}  noisy_mse={noisy_mse:.5f}  "
              f"val_total={val_total:.4f}  mse_improve=+{improve_pct:.1f}%")
        print(f"         feat_align={val_metrics['val_feat_align']:.5f}  "
              f"dist_align={val_metrics['val_dist_align']:.4f}  "
              f"mrd_fm={val_metrics['val_mrd_fm']:.4f}  "
              f"hubert={val_metrics['val_hubert']:.4f}  "
              f"stft_sc={val_metrics['val_stft_sc']:.4f}  "
              f"mel={val_metrics['val_mel_loss']:.4f}")
        print(f"  LR={current_lr:.3e}")

        if val_total < best_val_total:
            best_val_total = val_total
            torch.save({
                'epoch': epoch,
                'encoder_lora_state': {
                    k: v for k, v in model.student.state_dict().items() if 'lora_' in k
                },
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, exp_dir / 'best_model_val_total.pt')
            print(f"  New best val_total: {best_val_total:.4f} → saved best_model_val_total.pt")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                'epoch': epoch,
                'encoder_lora_state': {
                    k: v for k, v in model.student.state_dict().items() if 'lora_' in k
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
                    k: v for k, v in model.student.state_dict().items() if 'lora_' in k
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

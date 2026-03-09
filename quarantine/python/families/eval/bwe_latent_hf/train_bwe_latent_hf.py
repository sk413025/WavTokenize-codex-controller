"""
exp_0229c: Latent-domain BWE with HF-emphasis loss — 高頻精細化版本

改進自 exp_0229b：
    1. λ_wav 1.0 → 0.1（降低 MSE 主導，因為 MSE 懲罰相位誤差，壓制高頻學習）
    2. 新增 HF-emphasis STFT Loss（λ_hf=5.0，cutoff=4kHz，5x 高頻加重）
    3. 模型容量增大：hidden 128→256, kernel 3→5, blocks 6→8（~2.8M params）
    4. 新增 Multi-scale Mel Loss（在 n_fft=512,1024,2048 三個解析度計算）
    5. 評估增加高頻專屬指標：HF STFT loss, HF energy ratio

架構（與 exp_0229b 相同）：
    noisy → [Frozen Encoder LoRA (exp_0227)] → encoder_out [B,512,T]
                                                     ↓
                                    [Trainable LatentBWE v2]  ← 本實驗訓練目標
                                    dilated ResBlock × 8, hidden=256
                                    enhanced_latent [B,512,T]
                                                     ↓
                              ┌──────────────────────┴────────────────────┐
                              ↓ (論文架構)                                  ↓ (音質訓練)
                   [VQ → discrete token]                    [Frozen Decoder backbone+head]
                   token 品質同步提升                              enhanced_wav

Loss 設計：
    L = 0.1×MSE + 1.0×MR-STFT + 45.0×MultiMel + 2.0×MRD_FM + 5.0×HF-STFT

    HF-STFT：在 STFT domain 中，4kHz 以上頻率加權 5x，
    直接監督 LatentBWE 在高頻的輸出品質。

執行：
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    python families/eval/bwe_latent_hf/train_bwe_latent_hf.py --mode smoke

    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    python families/eval/bwe_latent_hf/train_bwe_latent_hf.py --mode epoch --epochs 300 --device cuda:1
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

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from families.deps.no_vq_core.models_no_vq import TeacherStudentNoVQ
from families.deps.encoder_aug.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum
from decoder.discriminators import MultiResolutionDiscriminator

SAMPLE_RATE = 24000

# exp_0227 best checkpoint
EXP0227_BEST_CKPT = (
    'families/deps/enc_mrd_fm/runs/enc_mrd_fm_epoch_20260227_024953/best_model_val_total.pt'
)


# ============================================================
# LatentBWE v2：增大模型容量
# ============================================================

class LatentResBlock(nn.Module):
    """Dilated ResBlock for latent-domain BWE。

    使用 dilated Conv1d 擴大感受野，讓每個 latent frame
    能利用鄰近 frames 的時序上下文推斷高頻 latent 分布。

    Args:
        channels: 隱藏通道數。
        kernel_size: 卷積核大小。
        dilation: 膨脹率。
    """

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, channels // 8), channels)
        self.norm2 = nn.GroupNorm(min(8, channels // 8), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播。

        Args:
            x: 輸入張量 [B, C, T]。

        Returns:
            殘差連接後的輸出 [B, C, T]。
        """
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual


class LatentBWE(nn.Module):
    """Latent-domain Bandwidth Extender v2。

    相比 v1（exp_0229b），增大模型容量以學習更精細的高頻 latent 映射。

    設計原則：
    1. 輸入輸出維度與 encoder output 相同（512-dim）
    2. 殘差學習：enhanced = encoder_out + Δ
    3. Dilated Conv1d 擴大感受野
    4. hidden_dim=256（v1 為 128），8 blocks（v1 為 6），kernel=5（v1 為 3）

    Args:
        latent_dim: encoder output 的通道數（WavTokenizer = 512）。
        hidden_dim: BWE 內部隱藏維度。
        num_blocks: ResBlock 數量。
        kernel_size: 卷積核大小。
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        num_blocks: int = 8,
        kernel_size: int = 5,
    ):
        super().__init__()

        # 投影到 hidden_dim
        self.proj_in = nn.Conv1d(latent_dim, hidden_dim, 1)

        # 多尺度 dilated ResBlocks（8 blocks: dilation=[1,2,4,8,1,2,4,8]）
        dilations = [1, 2, 4, 8, 1, 2, 4, 8][:num_blocks]
        self.blocks = nn.ModuleList([
            LatentResBlock(hidden_dim, kernel_size, dilation=d)
            for d in dilations
        ])

        # 投影回 latent_dim
        self.proj_out = nn.Conv1d(hidden_dim, latent_dim, 1)

        # 初始化 proj_out 為極小值 → 初始 Δ ≈ 0（不破壞 encoder 結果）
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.proj_out.bias)

        total = sum(p.numel() for p in self.parameters())
        print(f"  [LatentBWE-v2] {total:,} params "
              f"(latent={latent_dim}, hidden={hidden_dim}, "
              f"blocks={num_blocks}, kernel={kernel_size})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播。

        Args:
            x: encoder output [B, 512, T]。

        Returns:
            enhanced latent [B, 512, T]（殘差加法後）。
        """
        residual = x
        h = self.proj_in(x)           # [B, hidden, T]
        for block in self.blocks:
            h = block(h)
        delta = self.proj_out(h)       # [B, 512, T]
        return residual + delta


# ============================================================
# Frozen MRD Feature Matching Loss
# ============================================================

class FrozenMRDFeatureMatchingLoss(nn.Module):
    """使用凍結的預訓練 MRD 計算 feature matching loss。

    Args:
        wavtok_ckpt_path: WavTokenizer 預訓練權重路徑。
        device: 計算設備。
    """

    def __init__(self, wavtok_ckpt_path: str, device: torch.device):
        super().__init__()
        self.mrd = MultiResolutionDiscriminator(
            resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)),
        )
        ckpt = torch.load(wavtok_ckpt_path, map_location='cpu', weights_only=False)
        mrd_state = {
            k[len('multiresddisc.'):]: v
            for k, v in ckpt['state_dict'].items()
            if k.startswith('multiresddisc.')
        }
        missing, unexpected = self.mrd.load_state_dict(mrd_state, strict=False)
        if missing:    print(f"  [MRD] Missing keys: {len(missing)}")
        if unexpected: print(f"  [MRD] Unexpected keys: {len(unexpected)}")
        print(f"  [MRD] Loaded pretrained weights "
              f"({sum(p.numel() for p in self.mrd.parameters())/1e6:.2f}M params)")
        for p in self.mrd.parameters():
            p.requires_grad_(False)
        self.mrd.eval()
        total = sum(p.numel() for p in self.mrd.parameters())
        trainable = sum(p.numel() for p in self.mrd.parameters() if p.requires_grad)
        assert trainable == 0
        print(f"  [MRD] Frozen: {total:,} params, trainable: {trainable}")

    def forward(self, recon_wav: torch.Tensor, clean_wav: torch.Tensor) -> torch.Tensor:
        """計算 MRD feature matching loss。

        Args:
            recon_wav: 重建波形。
            clean_wav: 乾淨波形。

        Returns:
            Feature matching L1 loss。
        """
        if recon_wav.dim() == 3: recon_wav = recon_wav.squeeze(1)
        if clean_wav.dim() == 3: clean_wav = clean_wav.squeeze(1)
        with torch.no_grad():
            _, _, fmap_rs, _ = self.mrd(clean_wav, recon_wav.detach())
        _, _, _, fmap_gs = self.mrd(clean_wav.detach(), recon_wav)
        loss, n = 0.0, 0
        for fr_disc, fg_disc in zip(fmap_rs, fmap_gs):
            for fr, fg in zip(fr_disc, fg_disc):
                loss = loss + F.l1_loss(fg, fr.detach())
                n += 1
        return loss / max(n, 1)


# ============================================================
# Loss Functions
# ============================================================

class STFTLoss(nn.Module):
    """單一解析度 STFT 損失。

    Args:
        n_fft: FFT 大小。
        hop_length: 跳步長度。
        win_length: 視窗長度。
    """

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def _stft(self, x):
        """計算 STFT。

        Args:
            x: 輸入波形 [B, 1, T] 或 [B, T]。

        Returns:
            STFT 複數結果。
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        return torch.stft(x, self.n_fft, self.hop_length, self.win_length,
                          return_complex=True)

    def forward(self, y_hat, y):
        """計算 SC + Log Magnitude 損失。

        Args:
            y_hat: 預測波形。
            y: 目標波形。

        Returns:
            (sc_loss, log_mag_loss) 元組。
        """
        Y = self._stft(y)
        Y_hat = self._stft(y_hat)
        mag = Y.abs().clamp(min=1e-7)
        mag_hat = Y_hat.abs().clamp(min=1e-7)
        sc = torch.norm(mag - mag_hat, p='fro') / (torch.norm(mag, p='fro') + 1e-7)
        mag_loss = F.l1_loss(mag_hat.log(), mag.log())
        return sc, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """多解析度 STFT 損失。

    Args:
        fft_sizes: FFT 大小列表。
        hop_sizes: 跳步長度列表。
        win_sizes: 視窗長度列表。
    """

    def __init__(self, fft_sizes, hop_sizes, win_sizes):
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(n, h, w) for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, y_hat, y):
        """計算多解析度 STFT 損失。

        Args:
            y_hat: 預測波形。
            y: 目標波形。

        Returns:
            (mean_sc_loss, mean_mag_loss) 元組。
        """
        sc_loss, mag_loss = 0.0, 0.0
        for stft in self.stft_losses:
            sc, mag = stft(y_hat, y)
            sc_loss += sc
            mag_loss += mag
        return sc_loss / len(self.stft_losses), mag_loss / len(self.stft_losses)


class MelReconstructionLoss(nn.Module):
    """Mel 頻譜重建損失。

    Args:
        sample_rate: 取樣率。
        n_fft: FFT 大小。
        hop_length: 跳步長度。
        n_mels: Mel 頻帶數量。
    """

    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, center=True, power=1,
        )

    def forward(self, y_hat, y):
        """計算 Mel 頻譜 L1 損失。

        Args:
            y_hat: 預測波形 [B, T] 或 [B, 1, T]。
            y: 目標波形。

        Returns:
            Mel L1 loss。
        """
        if y_hat.dim() == 3: y_hat = y_hat.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)
        mel_hat = torch.log(self.mel_spec(y_hat).clamp(min=1e-7))
        mel = torch.log(self.mel_spec(y).clamp(min=1e-7))
        return F.l1_loss(mel_hat, mel)


class MultiScaleMelLoss(nn.Module):
    """多尺度 Mel 損失。

    在不同 n_fft 解析度下計算 Mel loss，捕捉不同時頻特徵。
    小 n_fft → 時間解析度高（暫態），大 n_fft → 頻率解析度高（穩態）。

    Args:
        sample_rate: 取樣率。
        n_fft_list: FFT 大小列表。
        hop_ratio: hop_length = n_fft // hop_ratio。
        n_mels: Mel 頻帶數量。
    """

    def __init__(self, sample_rate: int = 24000,
                 n_fft_list: List[int] = [512, 1024, 2048],
                 hop_ratio: int = 4, n_mels: int = 100):
        super().__init__()
        self.mel_losses = nn.ModuleList([
            MelReconstructionLoss(sample_rate, n_fft, n_fft // hop_ratio, n_mels)
            for n_fft in n_fft_list
        ])
        print(f"  [MultiScaleMel] {len(n_fft_list)} scales: "
              f"n_fft={n_fft_list}")

    def forward(self, y_hat, y):
        """計算多尺度 Mel 損失平均。

        Args:
            y_hat: 預測波形。
            y: 目標波形。

        Returns:
            多尺度 Mel L1 loss 平均值。
        """
        loss = 0.0
        for mel_loss in self.mel_losses:
            loss += mel_loss(y_hat, y)
        return loss / len(self.mel_losses)


class HighFreqEmphasisSTFTLoss(nn.Module):
    """高頻加重 STFT 損失。

    在 STFT domain 中，對 cutoff_hz 以上的頻率加重損失權重，
    鼓勵模型專注於學習缺失的高頻內容。

    Args:
        n_fft: FFT 大小。
        hop_length: 跳步長度。
        cutoff_hz: 高低頻分界頻率（Hz）。
        sr: 取樣率。
        hf_weight: 高頻區域的權重倍數。
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512,
                 cutoff_hz: float = 4000, sr: int = 24000, hf_weight: float = 5.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

        freq_bins = n_fft // 2 + 1
        weights = torch.ones(freq_bins)
        cutoff_bin = int(cutoff_hz / (sr / 2) * (freq_bins - 1))
        weights[cutoff_bin:] = hf_weight
        # [1, F, 1] for broadcasting with [B, F, T]
        self.register_buffer('freq_weights', weights.unsqueeze(0).unsqueeze(-1))

        self.cutoff_bin = cutoff_bin
        print(f"  [HF-STFT] cutoff={cutoff_hz}Hz (bin {cutoff_bin}/{freq_bins}), "
              f"hf_weight={hf_weight}x")

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """計算高頻加重 STFT 損失。

        Args:
            y_hat: 預測波形 [B, T] 或 [B, 1, T]。
            y: 目標波形。

        Returns:
            加權後的 STFT L1 損失。
        """
        if y_hat.dim() == 3: y_hat = y_hat.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)

        spec_hat = torch.stft(
            y_hat, self.n_fft, self.hop_length, window=self.window,
            return_complex=True,
        )
        spec = torch.stft(
            y, self.n_fft, self.hop_length, window=self.window,
            return_complex=True,
        )

        mag_hat = torch.abs(spec_hat)  # [B, F, T]
        mag = torch.abs(spec)

        # 加權後 L1 差異
        weighted_diff = self.freq_weights * torch.abs(mag_hat - mag)
        return weighted_diff.mean()

    def hf_only_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """僅計算高頻部分的 STFT L1 損失（用於評估）。

        Args:
            y_hat: 預測波形。
            y: 目標波形。

        Returns:
            4kHz 以上頻率的 STFT L1 損失。
        """
        if y_hat.dim() == 3: y_hat = y_hat.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)

        spec_hat = torch.stft(
            y_hat, self.n_fft, self.hop_length, window=self.window,
            return_complex=True,
        )
        spec = torch.stft(
            y, self.n_fft, self.hop_length, window=self.window,
            return_complex=True,
        )

        mag_hat = torch.abs(spec_hat)[:, self.cutoff_bin:, :]
        mag = torch.abs(spec)[:, self.cutoff_bin:, :]
        return F.l1_loss(mag_hat, mag)


# ============================================================
# Pipeline：Frozen Encoder → LatentBWE v2 → Frozen Decoder
# ============================================================

class LatentBWEPipeline(nn.Module):
    """完整推論 pipeline。

    noisy → frozen_encoder_lora → encoder_out [B,512,T]
                                       ↓
                               trainable LatentBWE v2
                                       ↓
                               enhanced_latent [B,512,T]
                                       ↓
                          ┌────────────┴──────────────┐
                          ↓                           ↓
                 VQ → discrete token          frozen_decoder
                 (for downstream LLM)      enhanced_wav [B,T]

    Args:
        wavtok_config: WavTokenizer 設定檔路徑。
        wavtok_ckpt: WavTokenizer 預訓練權重路徑。
        lora_rank: LoRA 秩。
        lora_alpha: LoRA alpha。
        bwe_hidden_dim: BWE 隱藏維度。
        bwe_num_blocks: BWE ResBlock 數量。
        bwe_kernel_size: BWE 卷積核大小。
        device: 計算設備。
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        bwe_hidden_dim: int = 256,
        bwe_num_blocks: int = 8,
        bwe_kernel_size: int = 5,
        device: torch.device = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device('cpu')

        # 建立 TeacherStudentNoVQ
        self.base_model = TeacherStudentNoVQ(
            wavtok_config=wavtok_config,
            wavtok_ckpt=wavtok_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            intermediate_indices=[3, 4, 6],
            device=device,
        )

        # 凍結整個 base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        frozen = sum(p.numel() for p in self.base_model.parameters())
        print(f"  [Pipeline] Base model frozen: {frozen:,} params")

        # LatentBWE v2
        self.bwe = LatentBWE(
            latent_dim=512,
            hidden_dim=bwe_hidden_dim,
            num_blocks=bwe_num_blocks,
            kernel_size=bwe_kernel_size,
        )
        trainable = sum(p.numel() for p in self.bwe.parameters())
        print(f"  [Pipeline] LatentBWE trainable: {trainable:,} params")

    def forward(
        self,
        clean_audio: torch.Tensor,
        noisy_audio: torch.Tensor,
    ) -> Dict:
        """前向傳播。

        Args:
            clean_audio: 乾淨音訊 [B, 1, T]。
            noisy_audio: 噪音音訊 [B, 1, T]。

        Returns:
            包含重建波形、增強 latent 等的字典。
        """
        with torch.no_grad():
            teacher_out, _ = self.base_model.teacher_extractor(clean_audio)
            encoder_out, _ = self.base_model.student_extractor(noisy_audio)
            recon_before = self.base_model.decode_continuous(encoder_out)

        # Trainable LatentBWE（梯度從這裡流）
        enhanced_latent = self.bwe(encoder_out)

        # Frozen decoder（讓梯度流回 enhanced_latent）
        bandwidth_id = torch.tensor([0], device=clean_audio.device)
        x = self.base_model.teacher.backbone(enhanced_latent, bandwidth_id=bandwidth_id)
        recon_wav = self.base_model.teacher.head(x)
        if recon_wav.dim() == 1:
            recon_wav = recon_wav.unsqueeze(0).unsqueeze(0)
        elif recon_wav.dim() == 2:
            recon_wav = recon_wav.unsqueeze(1)

        return {
            'recon_wav': recon_wav,
            'recon_wav_before_bwe': recon_before,
            'enhanced_latent': enhanced_latent,
            'encoder_out': encoder_out,
            'teacher_encoder_out': teacher_out,
        }

    def get_discrete_tokens(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """推論：取出 enhanced_latent 通過 VQ 的離散 token。

        Args:
            noisy_audio: 噪音音訊 [B, 1, T]。

        Returns:
            codes: token indices。
        """
        with torch.no_grad():
            encoder_out, _ = self.base_model.student_extractor(noisy_audio)
        enhanced = self.bwe(encoder_out)
        vq_out = self.base_model.vq(enhanced)
        return vq_out['codes']


# ============================================================
# Logging Utilities
# ============================================================

class _TeeIO:
    """同時輸出到多個串流。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        """寫入所有串流。"""
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        """刷新所有串流。"""
        for s in self.streams:
            s.flush()

    def isatty(self):
        """回傳 False。"""
        return False


def setup_logging(output_dir: Path) -> Path:
    """設定日誌輸出到檔案。

    Args:
        output_dir: 輸出目錄路徑。

    Returns:
        日誌檔案路徑。
    """
    import sys as _sys
    log_path = output_dir / 'train.log'
    log_f = open(log_path, 'w', encoding='utf-8', buffering=1)
    _sys.stdout = _TeeIO(_sys.__stdout__, log_f)
    _sys.stderr = _TeeIO(_sys.__stderr__, log_f)
    atexit.register(lambda: log_f.close())
    return log_path


def set_seed(seed: int = 42):
    """固定隨機種子。

    Args:
        seed: 隨機種子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_preinit(device: torch.device, retries: int = 10, sleep_s: float = 2.0):
    """預初始化 CUDA 裝置。

    Args:
        device: CUDA 裝置。
        retries: 最大重試次數。
        sleep_s: 重試間隔秒數。
    """
    import time as _time
    for i in range(retries):
        try:
            t = torch.zeros(1, device=device)
            del t
            print(f"CUDA pre-init OK (attempt {i+1})")
            return
        except RuntimeError:
            if i < retries - 1:
                _time.sleep(sleep_s)
    raise RuntimeError(f"CUDA pre-init failed after {retries} attempts")


# ============================================================
# Data Loaders
# ============================================================

def make_val_loader(val_cache_path, batch_size=4, num_workers=2):
    """建立驗證資料載入器。

    Args:
        val_cache_path: 驗證資料快取路徑。
        batch_size: 批次大小。
        num_workers: 工作行程數。

    Returns:
        DataLoader 物件。
    """
    ds = AugmentedCurriculumDataset(
        val_cache_path, augment=False,
        filter_clean_to_clean=True, compute_snr=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn_curriculum)


def make_train_loader(train_cache_path, batch_size=8, num_workers=2,
                      snr_remix_prob=0.5, snr_remix_range=(-5.0, 25.0),
                      random_gain_prob=0.3, random_gain_db=3.0,
                      random_crop_prob=0.3, random_crop_min_ratio=0.7,
                      time_stretch_prob=0.2, time_stretch_range=(0.95, 1.05)):
    """建立訓練資料載入器。

    Args:
        train_cache_path: 訓練資料快取路徑。
        batch_size: 批次大小。
        num_workers: 工作行程數。
        snr_remix_prob: SNR 重新混合機率。
        snr_remix_range: SNR 範圍。
        random_gain_prob: 隨機增益機率。
        random_gain_db: 隨機增益 dB 範圍。
        random_crop_prob: 隨機裁切機率。
        random_crop_min_ratio: 最小裁切比例。
        time_stretch_prob: 時間伸縮機率。
        time_stretch_range: 時間伸縮範圍。

    Returns:
        DataLoader 物件。
    """
    ds = AugmentedCurriculumDataset(
        train_cache_path, augment=True,
        filter_clean_to_clean=True, compute_snr=False,
        snr_remix_prob=snr_remix_prob, snr_remix_range=snr_remix_range,
        random_gain_prob=random_gain_prob, random_gain_db=random_gain_db,
        random_crop_prob=random_crop_prob, random_crop_min_ratio=random_crop_min_ratio,
        time_stretch_prob=time_stretch_prob, time_stretch_range=time_stretch_range,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, collate_fn=collate_fn_curriculum)


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(pipeline, mrd_fm_fn, dataloader, optimizer, device, epoch, config,
                mr_stft_fn, mel_fn, hf_stft_fn, scaler=None) -> Dict:
    """訓練一個 epoch。

    Args:
        pipeline: LatentBWE Pipeline 模型。
        mrd_fm_fn: MRD feature matching 損失函數。
        dataloader: 訓練資料載入器。
        optimizer: 優化器。
        device: 計算設備。
        epoch: 目前 epoch 數。
        config: 設定字典。
        mr_stft_fn: 多解析度 STFT 損失函數。
        mel_fn: Mel 損失函數（MultiScaleMel）。
        hf_stft_fn: 高頻加重 STFT 損失函數。
        scaler: GradScaler（AMP 用）。

    Returns:
        訓練指標字典。
    """
    pipeline.train()
    pipeline.base_model.eval()

    metrics = {k: 0.0 for k in [
        'total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
        'mel_loss', 'mrd_fm_loss', 'hf_stft_loss', 'nan_batches',
    ]}
    n_batches, nan_count = 0, 0

    lw  = config['lambda_wav']
    ls  = config['lambda_stft']
    lm  = config['lambda_mel']
    lfm = config['lambda_fm']
    lhf = config['lambda_hf']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [LatentBWE-v2]")

    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if clean.dim() == 2: clean = clean.unsqueeze(1)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

        if batch_idx % config['grad_accum'] == 0:
            optimizer.zero_grad()

        with autocast(enabled=config['use_amp']):
            out = pipeline(clean, noisy)
            recon = out['recon_wav']

            T = min(clean.shape[-1], recon.shape[-1])
            recon_t = recon[..., :T]
            clean_t = clean[..., :T]

            wav_mse  = F.mse_loss(recon_t, clean_t)
            sc, mag  = mr_stft_fn(recon_t, clean_t)
            mel_loss = mel_fn(recon_t, clean_t)
            mrd_fm   = mrd_fm_fn(recon_t, clean_t)
            hf_loss  = hf_stft_fn(recon_t, clean_t)

            loss = (lw * wav_mse
                    + ls * (sc + mag)
                    + lm * mel_loss
                    + lfm * mrd_fm
                    + lhf * hf_loss)
            loss = loss / config['grad_accum']

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            metrics['nan_batches'] = nan_count
            optimizer.zero_grad()
            if nan_count >= 10:
                print(f"  Too many NaN batches ({nan_count}), aborting epoch!")
                break
            continue

        if scaler:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in pipeline.bwe.parameters()],
                    max_norm=config['grad_clip'],
                )
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in pipeline.bwe.parameters()],
                    max_norm=config['grad_clip'],
                )
                optimizer.step()

        lv = loss.item() * config['grad_accum']
        metrics['total_loss']    += lv
        metrics['wav_mse']       += wav_mse.item()
        metrics['stft_sc_loss']  += sc.item()
        metrics['stft_mag_loss'] += mag.item()
        metrics['mel_loss']      += mel_loss.item()
        metrics['mrd_fm_loss']   += mrd_fm.item()
        metrics['hf_stft_loss']  += hf_loss.item()
        n_batches += 1

        pbar.set_postfix({
            'total':  f"{lv:.4f}",
            'wav':    f"{wav_mse.item():.5f}",
            'hf':     f"{hf_loss.item():.4f}",
        })

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
                  'mel_loss', 'mrd_fm_loss', 'hf_stft_loss']:
            metrics[k] /= n_batches
    return metrics


@torch.no_grad()
def evaluate(pipeline, mrd_fm_fn, dataloader, device, config,
             mr_stft_fn, mel_fn, hf_stft_fn, max_batches=30) -> Dict:
    """驗證。

    計算 BWE 前後的各項指標，包含高頻專屬指標。

    Args:
        pipeline: LatentBWE Pipeline 模型。
        mrd_fm_fn: MRD feature matching 損失函數。
        dataloader: 驗證資料載入器。
        device: 計算設備。
        config: 設定字典。
        mr_stft_fn: 多解析度 STFT 損失函數。
        mel_fn: Mel 損失函數。
        hf_stft_fn: 高頻加重 STFT 損失函數。
        max_batches: 最大評估批次數。

    Returns:
        驗證指標字典。
    """
    pipeline.eval()
    mse_list, noisy_mse_list = [], []
    sc_list, mag_list, mel_list, fm_list = [], [], [], []
    hf_list, hf_only_list = [], []
    mse_before_list = []
    # 高頻能量比
    hf_energy_ratio_before_list, hf_energy_ratio_after_list = [], []
    hf_energy_ratio_clean_list = []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if clean.dim() == 2: clean = clean.unsqueeze(1)
        if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

        out = pipeline(clean, noisy)
        recon  = out['recon_wav']
        before = out['recon_wav_before_bwe']

        T = min(clean.shape[-1], recon.shape[-1], noisy.shape[-1])
        clean_t = clean[..., :T]
        recon_t = recon[..., :T]
        noisy_t = noisy[..., :T]
        Tb = min(clean.shape[-1], before.shape[-1])
        before_t = before[..., :Tb]

        mse_list.append(F.mse_loss(recon_t, clean_t).item())
        mse_before_list.append(F.mse_loss(before_t, clean[..., :Tb]).item())
        noisy_mse_list.append(F.mse_loss(noisy_t, clean_t).item())

        sc, mag = mr_stft_fn(recon_t, clean_t)
        sc_list.append(sc.item())
        mag_list.append(mag.item())
        mel_list.append(mel_fn(recon_t, clean_t).item())
        fm_list.append(mrd_fm_fn(recon_t, clean_t).item())
        hf_list.append(hf_stft_fn(recon_t, clean_t).item())
        hf_only_list.append(hf_stft_fn.hf_only_loss(recon_t, clean_t).item())

        # 計算高頻能量比（>4kHz energy / total energy）
        def _hf_energy_ratio(wav):
            """計算 4kHz 以上的頻率能量佔比。"""
            if wav.dim() == 3: wav = wav.squeeze(1)
            spec = torch.stft(wav, 2048, 512, return_complex=True)
            mag_sq = torch.abs(spec) ** 2
            cutoff = hf_stft_fn.cutoff_bin
            hf_energy = mag_sq[:, cutoff:, :].sum()
            total_energy = mag_sq.sum() + 1e-10
            return (hf_energy / total_energy).item()

        hf_energy_ratio_before_list.append(_hf_energy_ratio(before_t))
        hf_energy_ratio_after_list.append(_hf_energy_ratio(recon_t))
        hf_energy_ratio_clean_list.append(_hf_energy_ratio(clean_t))

    pipeline.train()
    pipeline.base_model.eval()

    def _m(lst):
        """計算平均值。"""
        return float(np.mean(lst)) if lst else float('nan')

    return {
        'val_wav_mse':        _m(mse_list),
        'val_wav_mse_before': _m(mse_before_list),
        'val_noisy_mse':      _m(noisy_mse_list),
        'val_stft_sc':        _m(sc_list),
        'val_stft_mag':       _m(mag_list),
        'val_mel_loss':       _m(mel_list),
        'val_mrd_fm':         _m(fm_list),
        'val_hf_stft':        _m(hf_list),
        'val_hf_only':        _m(hf_only_list),
        'val_hf_ratio_before': _m(hf_energy_ratio_before_list),
        'val_hf_ratio_after':  _m(hf_energy_ratio_after_list),
        'val_hf_ratio_clean':  _m(hf_energy_ratio_clean_list),
    }


def _save_audio_samples(pipeline, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    """儲存音訊樣本。

    Args:
        pipeline: Pipeline 模型。
        loader: 資料載入器。
        device: 計算設備。
        output_dir: 輸出目錄。
        epoch: 目前 epoch。
        num_samples: 儲存樣本數。
        split: 資料集分割名稱。
    """
    audio_dir = output_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)
    pipeline.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            if saved >= num_samples:
                break
            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            if clean.dim() == 2: clean = clean.unsqueeze(1)
            if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

            out = pipeline(clean, noisy)
            recon  = out['recon_wav']
            before = out['recon_wav_before_bwe']
            B = min(noisy.shape[0], num_samples - saved)
            for b in range(B):
                def _save(t, name):
                    """儲存單一波形為 WAV 檔。"""
                    w = t[b].squeeze().cpu().float().numpy()
                    w = np.clip(w, -1.0, 1.0)
                    wavfile.write(str(audio_dir / name), SAMPLE_RATE,
                                  (w * 32767).astype(np.int16))
                _save(noisy, f'sample{saved+b:02d}_noisy.wav')
                T  = min(clean.shape[-1], recon.shape[-1])
                Tb = min(clean.shape[-1], before.shape[-1])
                _save(recon[..., :T],   f'sample{saved+b:02d}_recon_after_bwe.wav')
                _save(before[..., :Tb], f'sample{saved+b:02d}_recon_before_bwe.wav')
                _save(clean[..., :T],   f'sample{saved+b:02d}_clean.wav')
            saved += B
    pipeline.train()
    pipeline.base_model.eval()
    print(f"  Audio saved ({split}) → {audio_dir}")


def _save_spectrogram_comparison(pipeline, loader, device, output_dir, epoch,
                                 cutoff_hz=4000, num_samples=2):
    """儲存頻譜圖比較（BWE 前 vs BWE 後 vs Clean）。

    Args:
        pipeline: Pipeline 模型。
        loader: 資料載入器。
        device: 計算設備。
        output_dir: 輸出目錄。
        epoch: 目前 epoch。
        cutoff_hz: 高頻分界線頻率（紅色虛線）。
        num_samples: 儲存樣本數。
    """
    spec_dir = output_dir / 'spectrograms' / f'epoch_{epoch:03d}'
    spec_dir.mkdir(parents=True, exist_ok=True)
    pipeline.eval()
    saved = 0

    with torch.no_grad():
        for batch in loader:
            if saved >= num_samples:
                break
            noisy = batch['noisy_audio'].to(device)
            clean = batch['clean_audio'].to(device)
            if clean.dim() == 2: clean = clean.unsqueeze(1)
            if noisy.dim() == 2: noisy = noisy.unsqueeze(1)

            out = pipeline(clean, noisy)
            recon  = out['recon_wav']
            before = out['recon_wav_before_bwe']

            B = min(noisy.shape[0], num_samples - saved)
            for b in range(B):
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                T  = min(clean.shape[-1], recon.shape[-1])
                Tb = min(clean.shape[-1], before.shape[-1])

                wavs = [
                    before[b].squeeze().cpu().float().numpy()[:Tb],
                    recon[b].squeeze().cpu().float().numpy()[:T],
                    clean[b].squeeze().cpu().float().numpy()[:T],
                ]
                titles = [
                    'Before BWE (exp_0227)',
                    'After BWE (exp_0229c)',
                    'Clean',
                ]

                for ax, wav, title in zip(axes, wavs, titles):
                    spec = np.abs(np.fft.rfft(
                        np.lib.stride_tricks.sliding_window_view(wav, 2048)[::512]
                    ))
                    spec_db = 20 * np.log10(spec.T + 1e-7)
                    im = ax.imshow(spec_db, origin='lower', aspect='auto',
                                   cmap='magma', vmin=-60, vmax=0,
                                   extent=[0, len(wav)/SAMPLE_RATE,
                                           0, SAMPLE_RATE/2])
                    ax.axhline(cutoff_hz, color='red', ls='--', linewidth=1,
                               alpha=0.8, label=f'{cutoff_hz}Hz')
                    ax.set_title(title, fontsize=11)
                    ax.set_ylabel('Frequency (Hz)')
                    ax.set_xlabel('Time (s)')
                    ax.legend(loc='upper right', fontsize=8)

                fig.suptitle(f'Spectrogram Comparison — Epoch {epoch}, Sample {saved+b}',
                             fontsize=13)
                plt.tight_layout()
                plt.savefig(spec_dir / f'sample{saved+b:02d}_spec_compare.png', dpi=120)
                plt.close()
            saved += B

    pipeline.train()
    pipeline.base_model.eval()
    print(f"  Spectrograms saved → {spec_dir}")


def plot_training_curves(history, output_dir, epoch):
    """繪製訓練曲線。

    Args:
        history: 訓練歷史字典。
        output_dir: 輸出目錄。
        epoch: 目前 epoch。
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f'exp_0229c: LatentBWE-v2 + HF-emphasis — Epoch {epoch}', fontsize=14)
    epochs = range(1, len(history['train_total_loss']) + 1)

    # (0,0) Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train total', alpha=0.8)
    if history.get('val_total'):
        ax.plot(epochs, history['val_total'], 'r--', label='Val total', alpha=0.8)
    ax.set_title('Total Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    # (0,1) MSE: before vs after BWE
    ax = axes[0, 1]
    if history.get('val_noisy_mse'):
        ax.plot(epochs, history['val_noisy_mse'], 'gray', ls=':', label='Noisy baseline')
    if history.get('val_wav_mse_before'):
        ax.plot(epochs, history['val_wav_mse_before'], 'orange', ls='--', label='Before BWE')
    if history.get('val_wav_mse'):
        ax.plot(epochs, history['val_wav_mse'], 'r-', label='After BWE')
    ax.set_title('Wav MSE (val): Before vs After BWE')
    ax.legend()
    ax.grid(True)

    # (1,0) MRD FM
    ax = axes[1, 0]
    if history.get('train_mrd_fm'):
        ax.plot(epochs, history['train_mrd_fm'], 'orange', label='Train MRD FM', alpha=0.8)
    if history.get('val_mrd_fm'):
        ax.plot(epochs, history['val_mrd_fm'], 'red', ls='--', label='Val MRD FM')
    ax.set_title('MRD Feature Matching Loss')
    ax.legend()
    ax.grid(True)

    # (1,1) HF-specific losses
    ax = axes[1, 1]
    if history.get('train_hf_stft'):
        ax.plot(epochs, history['train_hf_stft'], 'b-', label='Train HF-STFT', alpha=0.8)
    if history.get('val_hf_stft'):
        ax.plot(epochs, history['val_hf_stft'], 'r--', label='Val HF-STFT')
    if history.get('val_hf_only'):
        ax.plot(epochs, history['val_hf_only'], 'purple', ls=':', label='Val HF-only (>4kHz)')
    ax.set_title('High-Frequency Losses')
    ax.legend()
    ax.grid(True)

    # (2,0) HF Energy Ratio
    ax = axes[2, 0]
    if history.get('val_hf_ratio_clean'):
        ax.plot(epochs, history['val_hf_ratio_clean'], 'g-', label='Clean HF ratio', alpha=0.6)
    if history.get('val_hf_ratio_before'):
        ax.plot(epochs, history['val_hf_ratio_before'], 'orange', ls='--', label='Before BWE')
    if history.get('val_hf_ratio_after'):
        ax.plot(epochs, history['val_hf_ratio_after'], 'r-', label='After BWE')
    ax.set_title('HF Energy Ratio (>4kHz / total)')
    ax.set_ylabel('Ratio')
    ax.legend()
    ax.grid(True)

    # (2,1) Spectral losses
    ax = axes[2, 1]
    if history.get('val_stft_sc'):
        ax.plot(epochs, history['val_stft_sc'], 'b-', label='STFT SC')
    if history.get('val_mel_loss'):
        ax.plot(epochs, history['val_mel_loss'], 'r--', label='Mel')
    ax.set_title('Spectral Losses (val)')
    ax.legend()
    ax.grid(True)

    # (3,0) LR
    ax = axes[3, 0]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'g-')
    ax.set_title('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True)

    # (3,1) BWE improvement %
    ax = axes[3, 1]
    if history.get('val_wav_mse') and history.get('val_wav_mse_before'):
        before_arr = np.array(history['val_wav_mse_before'])
        after_arr  = np.array(history['val_wav_mse'])
        improve_pct = (before_arr - after_arr) / (before_arr + 1e-10) * 100
        ax.plot(epochs, improve_pct, 'purple', label='MSE improve %')
        ax.axhline(0, color='black', ls='--', linewidth=0.8)
    ax.set_title('BWE MSE Improvement over Baseline (%)')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch{epoch:03d}.png', dpi=100)
    plt.close()
    print(f"  Loss plot saved → epoch {epoch}")


# ============================================================
# Main
# ============================================================

def main():
    """主函式：解析參數、建立模型、訓練與評估。"""
    parser = argparse.ArgumentParser(
        description='exp_0229c: Latent BWE with HF-emphasis loss'
    )
    parser.add_argument('--mode', type=str, default='smoke', choices=['smoke', 'epoch'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--encoder_ckpt', type=str, default=EXP0227_BEST_CKPT)

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

    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    # LatentBWE v2 超參數（增大模型容量）
    parser.add_argument('--bwe_hidden_dim', type=int, default=256,
                        help='BWE 隱藏維度（v1=128, v2=256）')
    parser.add_argument('--bwe_num_blocks', type=int, default=8,
                        help='ResBlock 數量（v1=6, v2=8）')
    parser.add_argument('--bwe_kernel_size', type=int, default=5,
                        help='卷積核大小（v1=3, v2=5）')

    # Loss 權重（核心改進）
    parser.add_argument('--lambda_wav',  type=float, default=0.1,
                        help='MSE 權重（v1=1.0, v2=0.1 — 降低相位懲罰）')
    parser.add_argument('--lambda_stft', type=float, default=1.0)
    parser.add_argument('--lambda_mel',  type=float, default=45.0)
    parser.add_argument('--lambda_fm',   type=float, default=2.0)
    parser.add_argument('--lambda_hf',   type=float, default=5.0,
                        help='HF-emphasis STFT 權重（新增）')

    # HF-emphasis STFT 設定
    parser.add_argument('--hf_cutoff_hz', type=float, default=4000.0,
                        help='高低頻分界頻率')
    parser.add_argument('--hf_weight', type=float, default=5.0,
                        help='高頻區域的權重倍數')

    parser.add_argument('--stft_fft_sizes', type=str, default='2048,1024,512')
    parser.add_argument('--stft_hop_sizes', type=str, default='512,256,128')
    parser.add_argument('--stft_win_sizes', type=str, default='2048,1024,512')

    parser.add_argument('--snr_remix_prob',    type=float, default=0.5)
    parser.add_argument('--snr_remix_min',     type=float, default=-5.0)
    parser.add_argument('--snr_remix_max',     type=float, default=25.0)
    parser.add_argument('--random_gain_prob',  type=float, default=0.3)
    parser.add_argument('--random_gain_db',    type=float, default=3.0)
    parser.add_argument('--random_crop_prob',  type=float, default=0.3)
    parser.add_argument('--random_crop_min_ratio', type=float, default=0.7)
    parser.add_argument('--time_stretch_prob', type=float, default=0.2)
    parser.add_argument('--time_stretch_min',  type=float, default=0.95)
    parser.add_argument('--time_stretch_max',  type=float, default=1.05)

    parser.add_argument('--save_checkpoint_every', type=int, default=10)
    parser.add_argument('--save_audio_interval',   type=int, default=25)
    parser.add_argument('--eval_max_batches',       type=int, default=30)

    args = parser.parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    BASE_DIR = Path(__file__).parent  # families/eval/bwe_latent_hf/
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    elif args.mode == 'smoke':
        exp_dir = BASE_DIR / f'runs/bwe_latent_hf_smoke_{timestamp}'
    else:
        exp_dir = BASE_DIR / f'runs/bwe_latent_hf_epoch_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    fft_sizes = [int(x) for x in args.stft_fft_sizes.split(',')]
    hop_sizes = [int(x) for x in args.stft_hop_sizes.split(',')]
    win_sizes = [int(x) for x in args.stft_win_sizes.split(',')]

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0229c_latent_bwe_hf'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0229c: Latent-domain BWE with HF-emphasis loss")
    print("=" * 70)
    print(f"Mode: {args.mode} | Epochs: {args.epochs} | Device: {args.device}")
    print(f"LR: {args.learning_rate} → {args.min_lr}")
    print(f"Encoder LoRA: FROZEN (from exp_0227)")
    print(f"Decoder: FROZEN | MRD: FROZEN")
    print(f"LatentBWE-v2: hidden={args.bwe_hidden_dim}, blocks={args.bwe_num_blocks}, "
          f"kernel={args.bwe_kernel_size}")
    print(f"Loss: λ_wav={args.lambda_wav} + λ_stft={args.lambda_stft} "
          f"+ λ_mel={args.lambda_mel} + λ_fm={args.lambda_fm} "
          f"+ λ_hf={args.lambda_hf} (HF-emphasis)")
    print(f"HF cutoff: {args.hf_cutoff_hz}Hz, weight: {args.hf_weight}x")
    print(f"Encoder ckpt: {args.encoder_ckpt}")
    print(f"Output: {exp_dir}")
    if log_path:
        print(f"Log: {log_path}")
    print("=" * 70)

    device = torch.device(args.device)
    cuda_preinit(device)

    # Data
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
        if args.epochs > 5:
            args.epochs = 5
            print(f"  Smoke mode: capping epochs to {args.epochs}")
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

    # Build pipeline
    print("\nBuilding LatentBWE-v2 Pipeline...")
    pipeline = LatentBWEPipeline(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        bwe_hidden_dim=args.bwe_hidden_dim,
        bwe_num_blocks=args.bwe_num_blocks,
        bwe_kernel_size=args.bwe_kernel_size,
        device=device,
    ).to(device)

    # Load frozen encoder LoRA from exp_0227
    ckpt_path = Path(args.encoder_ckpt)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            missing, unexpected = pipeline.base_model.load_state_dict(
                ckpt['model_state_dict'], strict=False,
            )
            ep = ckpt.get('epoch', '?')
            print(f"  Loaded exp_0227 model_state_dict (ep{ep})")
            if missing:    print(f"  Missing: {len(missing)}")
            if unexpected: print(f"  Unexpected: {len(unexpected)}")
        for p in pipeline.base_model.parameters():
            p.requires_grad_(False)
        frozen = sum(p.numel() for p in pipeline.base_model.parameters())
        print(f"  Pipeline frozen confirmed: {frozen:,} params")
    else:
        print(f"  WARNING: encoder ckpt not found: {ckpt_path}")

    # Frozen MRD
    print("\nLoading Frozen MRD discriminator...")
    mrd_fm_fn = FrozenMRDFeatureMatchingLoss(
        wavtok_ckpt_path=str(WAVTOK_CKPT), device=device,
    ).to(device)

    # Loss functions
    mr_stft_fn = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_sizes).to(device)
    mel_fn = MultiScaleMelLoss(
        sample_rate=SAMPLE_RATE, n_fft_list=[512, 1024, 2048],
    ).to(device)
    hf_stft_fn = HighFreqEmphasisSTFTLoss(
        n_fft=2048, hop_length=512,
        cutoff_hz=args.hf_cutoff_hz, sr=SAMPLE_RATE, hf_weight=args.hf_weight,
    ).to(device)

    # Optimizer
    bwe_params = list(pipeline.bwe.parameters())
    trainable_count = sum(p.numel() for p in bwe_params)
    total_count = sum(p.numel() for p in pipeline.parameters())
    print(f"\nTrainable (LatentBWE-v2): {trainable_count:,} / {total_count:,} "
          f"({100*trainable_count/total_count:.3f}%)")

    optimizer = torch.optim.AdamW(
        bwe_params, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    def lr_lambda(ep):
        """餘弦退火學習率排程。"""
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.learning_rate, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if args.use_amp else None

    best_val_total = float('inf')
    best_val_mse   = float('inf')
    history = {
        'train_total_loss': [], 'train_wav_mse': [],
        'train_stft_sc': [], 'train_stft_mag': [], 'train_mel': [],
        'train_mrd_fm': [], 'train_hf_stft': [],
        'val_wav_mse': [], 'val_wav_mse_before': [], 'val_noisy_mse': [],
        'val_stft_sc': [], 'val_stft_mag': [], 'val_mel_loss': [],
        'val_mrd_fm': [], 'val_hf_stft': [], 'val_hf_only': [],
        'val_hf_ratio_before': [], 'val_hf_ratio_after': [], 'val_hf_ratio_clean': [],
        'val_total': [],
        'lr': [],
    }

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_epoch(
            pipeline, mrd_fm_fn, train_loader, optimizer, device, epoch, config,
            mr_stft_fn, mel_fn, hf_stft_fn, scaler,
        )
        val_m = evaluate(
            pipeline, mrd_fm_fn, val_loader, device, config,
            mr_stft_fn, mel_fn, hf_stft_fn, args.eval_max_batches,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # History
        history['train_total_loss'].append(train_m['total_loss'])
        history['train_wav_mse'].append(train_m['wav_mse'])
        history['train_stft_sc'].append(train_m['stft_sc_loss'])
        history['train_stft_mag'].append(train_m['stft_mag_loss'])
        history['train_mel'].append(train_m['mel_loss'])
        history['train_mrd_fm'].append(train_m['mrd_fm_loss'])
        history['train_hf_stft'].append(train_m['hf_stft_loss'])
        history['val_wav_mse'].append(val_m['val_wav_mse'])
        history['val_wav_mse_before'].append(val_m['val_wav_mse_before'])
        history['val_noisy_mse'].append(val_m['val_noisy_mse'])
        history['val_stft_sc'].append(val_m['val_stft_sc'])
        history['val_stft_mag'].append(val_m['val_stft_mag'])
        history['val_mel_loss'].append(val_m['val_mel_loss'])
        history['val_mrd_fm'].append(val_m['val_mrd_fm'])
        history['val_hf_stft'].append(val_m['val_hf_stft'])
        history['val_hf_only'].append(val_m['val_hf_only'])
        history['val_hf_ratio_before'].append(val_m['val_hf_ratio_before'])
        history['val_hf_ratio_after'].append(val_m['val_hf_ratio_after'])
        history['val_hf_ratio_clean'].append(val_m['val_hf_ratio_clean'])
        history['lr'].append(lr_now)

        val_mse    = val_m['val_wav_mse']
        noisy_mse  = val_m['val_noisy_mse']
        before_mse = val_m['val_wav_mse_before']
        val_total = (val_mse
                     + val_m['val_stft_sc']
                     + val_m['val_stft_mag']
                     + args.lambda_mel * val_m['val_mel_loss']
                     + args.lambda_hf * val_m['val_hf_stft'])
        history['val_total'].append(val_total)

        improve_vs_before = (before_mse - val_mse) / before_mse * 100 if before_mse > 0 else 0

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: total={train_m['total_loss']:.4f}  "
              f"wav={train_m['wav_mse']:.5f}  "
              f"hf={train_m['hf_stft_loss']:.4f}  "
              f"mrd_fm={train_m['mrd_fm_loss']:.4f}")
        print(f"  Val:   mse_after={val_mse:.5f}  mse_before={before_mse:.5f}  "
              f"(BWE Δ={improve_vs_before:+.1f}%)  "
              f"noisy={noisy_mse:.5f}  total={val_total:.4f}")
        print(f"         stft_sc={val_m['val_stft_sc']:.4f}  "
              f"mel={val_m['val_mel_loss']:.4f}  "
              f"hf_stft={val_m['val_hf_stft']:.4f}  "
              f"hf_only={val_m['val_hf_only']:.4f}")
        print(f"         HF ratio: clean={val_m['val_hf_ratio_clean']:.4f}  "
              f"before={val_m['val_hf_ratio_before']:.4f}  "
              f"after={val_m['val_hf_ratio_after']:.4f}")
        print(f"  LR={lr_now:.3e}")

        save_dict = {
            'epoch': epoch,
            'bwe_state_dict': pipeline.bwe.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': val_m,
            'config': config,
        }

        if val_total < best_val_total:
            best_val_total = val_total
            torch.save(save_dict, exp_dir / 'best_model_val_total.pt')
            print(f"  ★ New best val_total: {best_val_total:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(save_dict, exp_dir / 'best_model.pt')
            print(f"  ★ New best val_mse: {best_val_mse:.5f}")

        if epoch % args.save_checkpoint_every == 0:
            torch.save(save_dict, exp_dir / f'checkpoint_epoch{epoch:03d}.pt')

        if epoch % args.save_audio_interval == 0 or epoch == args.epochs:
            plot_training_curves(history, exp_dir, epoch)
            _save_audio_samples(pipeline, val_loader, device, exp_dir, epoch,
                                num_samples=2, split='val')
            _save_audio_samples(pipeline, train_loader, device, exp_dir, epoch,
                                num_samples=2, split='train')
            _save_spectrogram_comparison(pipeline, val_loader, device, exp_dir, epoch,
                                         cutoff_hz=args.hf_cutoff_hz)

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_total: {best_val_total:.4f}, "
          f"best val_mse: {best_val_mse:.5f}")
    print(f"Output: {exp_dir}")


if __name__ == '__main__':
    main()

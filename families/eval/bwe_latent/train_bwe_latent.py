"""
exp_0229b: Latent-domain BWE — Option 1

架構（Option 1）：
    noisy → [Frozen Encoder LoRA (exp_0227)] → encoder_out [B,512,T]
                                                     ↓
                                          [Trainable LatentBWE]   ← 本實驗訓練目標
                                          dilated ResBlock × N
                                          enhanced_latent [B,512,T]
                                                     ↓
                              ┌──────────────────────┴────────────────────┐
                              ↓ (論文架構)                                  ↓ (音質訓練)
                   [VQ → discrete token]                    [Frozen Decoder backbone+head]
                   token 品質同步提升                              enhanced_wav
                              ↓
                      下游大型語言模型

與 exp_0229（Option 2, wav-domain BWE）的差異：
    - exp_0229:  frozen_encoder → frozen_decoder → recon_wav → trainable_BWE → enhanced_wav
                 token 不受 BWE 影響，論文架構不受益
    - exp_0229b: frozen_encoder → trainable_LatentBWE → enhanced_latent → frozen_decoder
                 token 來自 enhanced_latent（VQ on enhanced）→ token 品質提升
                 論文架構完全相容

LatentBWE 設計：
    - 輸入/輸出都是 [B, 512, T]（與 encoder output 相同維度）
    - 殘差學習：output = input + Δ（初始 Δ≈0，不破壞已有重建品質）
    - 多尺度 dilated Conv1d（dilation=1,2,4,8）擴大時域感受野
    - 每個 latent frame 能看到 ±若干 frame 的時序上下文
    - 讓 BWE 從「鄰近 frames 的低頻模式」推斷高頻 latent 應有的分布

訓練策略：
    - Encoder LoRA（exp_0227 best_model_val_total.pt ep161）完全凍結
    - Decoder 完全凍結
    - 只訓練 LatentBWE（~0.5M params）
    - Loss: λ_wav × MSE + λ_stft × MR-STFT + λ_mel × Mel + λ_fm × MRD_FM
           （與 0227 相同，但梯度目標改為 LatentBWE）

執行：
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python families/eval/bwe_latent/train_bwe_latent.py --mode smoke

    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python families/eval/bwe_latent/train_bwe_latent.py \\
        --mode epoch --epochs 200 --device cuda:1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# exp_0227 best_model_val_total.pt（ep161）
EXP0227_BEST_CKPT = (
    'families/deps/enc_mrd_fm/runs/enc_mrd_fm_epoch_20260227_024953/best_model_val_total.pt'
)

# ============================================================
# LatentBWE：在 latent 域（encoder output）做頻寬擴展
# ============================================================

class LatentResBlock(nn.Module):
    """Dilated ResBlock for latent-domain BWE.

    使用 dilated Conv1d 擴大感受野，讓每個 latent frame
    能利用鄰近 frames 的時序上下文推斷高頻 latent 分布。

    Args:
        channels: 隱藏通道數（等於 latent dim=512 或投影後的 dim）
        kernel_size: 卷積核大小
        dilation: 膨脹率
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, channels // 8), channels)
        self.norm2 = nn.GroupNorm(min(8, channels // 8), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual


class LatentBWE(nn.Module):
    """Latent-domain Bandwidth Extender.

    在 encoder output [B, 512, T] 的 latent 空間插入一個可訓練的
    residual network，學習從「低頻主導的 LDV latent」推斷「clean latent 應有的分布」。

    設計原則：
    1. 輸入輸出維度與 encoder output 相同（512-dim），確保後接的 frozen decoder 無需改動
    2. 殘差學習：enhanced = encoder_out + BWE(encoder_out)
       - 初始化讓 BWE output ≈ 0，訓練從 identity mapping 開始
       - 避免破壞 encoder 已學到的低頻重建能力
    3. Dilated Conv1d 擴大感受野，利用時序上下文
    4. 內部用 hidden_dim < 512 降維減少參數量（預設 128）

    Args:
        latent_dim: encoder output 的通道數（WavTokenizer = 512）
        hidden_dim: BWE 內部隱藏維度（影響參數量）
        num_blocks: ResBlock 數量
        kernel_size: 卷積核大小
    """
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 128,
        num_blocks: int = 6,
        kernel_size: int = 3,
    ):
        super().__init__()

        # 投影到 hidden_dim
        self.proj_in  = nn.Conv1d(latent_dim, hidden_dim, 1)

        # 多尺度 dilated ResBlocks
        dilations = [1, 2, 4, 8, 1, 2][:num_blocks]
        self.blocks = nn.ModuleList([
            LatentResBlock(hidden_dim, kernel_size, dilation=d)
            for d in dilations
        ])

        # 投影回 latent_dim
        self.proj_out = nn.Conv1d(hidden_dim, latent_dim, 1)

        # 初始化 proj_out 為零 → 初始輸出 ≈ identity（不破壞 encoder 結果）
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        total = sum(p.numel() for p in self.parameters())
        print(f"  [LatentBWE] {total:,} params "
              f"(latent={latent_dim}, hidden={hidden_dim}, "
              f"blocks={num_blocks}, kernel={kernel_size})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: encoder output [B, 512, T]
        Returns:
            enhanced latent [B, 512, T]  (same shape, residual added)
        """
        residual = x
        h = self.proj_in(x)           # [B, hidden, T]
        for block in self.blocks:
            h = block(h)              # [B, hidden, T]
        delta = self.proj_out(h)      # [B, 512, T]
        return residual + delta       # 殘差加法


# ============================================================
# Frozen MRD Feature Matching Loss（複製自 exp_0227）
# ============================================================

class FrozenMRDFeatureMatchingLoss(nn.Module):
    """使用凍結的預訓練 MRD 計算 feature matching loss。"""

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
# Standard Losses（與 exp_0227 相同）
# ============================================================

class STFTLoss(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def _stft(self, x):
        return torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length,
                          return_complex=True)

    def forward(self, y_hat, y):
        Y     = self._stft(y)
        Y_hat = self._stft(y_hat)
        mag     = Y.abs().clamp(min=1e-7)
        mag_hat = Y_hat.abs().clamp(min=1e-7)
        sc  = torch.norm(mag - mag_hat, p='fro') / (torch.norm(mag, p='fro') + 1e-7)
        mag_loss = F.l1_loss(mag_hat.log(), mag.log())
        return sc, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_sizes):
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(n, h, w) for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, y_hat, y):
        sc_loss, mag_loss = 0.0, 0.0
        for stft in self.stft_losses:
            sc, mag = stft(y_hat, y)
            sc_loss += sc; mag_loss += mag
        return sc_loss / len(self.stft_losses), mag_loss / len(self.stft_losses)


class MelReconstructionLoss(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        super().__init__()
        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        mel_f = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)
        mel_lo = 2595 * torch.log10(torch.tensor(1.0) + mel_f / 700)
        mel_pts = torch.linspace(mel_lo[0], mel_lo[-1], n_mels + 2)
        hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
        bins    = ((n_fft + 1) * hz_pts / sample_rate).long().clamp(0, n_fft // 2)
        for m in range(1, n_mels + 1):
            for k in range(bins[m-1], bins[m]):
                fb[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1] + 1e-8)
            for k in range(bins[m], bins[m+1]):
                fb[m-1, k] = (bins[m+1] - k) / (bins[m+1] - bins[m] + 1e-8)
        self.register_buffer('fb', fb)
        self.n_fft = n_fft; self.hop_length = hop_length

    def forward(self, y_hat, y):
        def mel(x):
            S = torch.stft(x.squeeze(1), self.n_fft, self.hop_length,
                           return_complex=True).abs().clamp(min=1e-7)
            return torch.log(torch.matmul(self.fb, S) + 1e-7)
        return F.l1_loss(mel(y_hat), mel(y))


# ============================================================
# Pipeline：Frozen Encoder → LatentBWE → Frozen Decoder
# ============================================================

class LatentBWEPipeline(nn.Module):
    """
    完整推論 pipeline（Option 1）：

        noisy → frozen_encoder_lora → encoder_out [B,512,T]
                                           ↓
                                   trainable LatentBWE
                                           ↓
                                   enhanced_latent [B,512,T]
                                           ↓
                          ┌────────────────┴──────────────┐
                          ↓                               ↓
                 VQ → discrete token              frozen_decoder
                 (for downstream LLM)          enhanced_wav [B,T]

    Encoder LoRA 完全凍結，只有 LatentBWE 接受梯度。
    """
    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        bwe_hidden_dim: int = 128,
        bwe_num_blocks: int = 6,
        bwe_kernel_size: int = 3,
        device: torch.device = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device('cpu')

        # 1. 建立 TeacherStudentNoVQ（encoder + decoder，No-VQ path）
        self.base_model = TeacherStudentNoVQ(
            wavtok_config=wavtok_config,
            wavtok_ckpt=wavtok_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            intermediate_indices=[3, 4, 6],
            device=device,
        )

        # 2. 凍結整個 base_model（encoder LoRA + decoder 全部）
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        frozen = sum(p.numel() for p in self.base_model.parameters())
        print(f"  [Pipeline] Base model frozen: {frozen:,} params")

        # 3. LatentBWE（唯一可訓練部分）
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
        """
        Args:
            clean_audio: [B, 1, T]
            noisy_audio: [B, 1, T]
        Returns:
            dict with keys:
                recon_wav_before_bwe: decoder(encoder_out)  — BWE 前（reference）
                recon_wav:            decoder(enhanced_latent) — BWE 後（訓練目標）
                enhanced_latent:      LatentBWE 輸出 [B, 512, T]
                encoder_out:          原始 encoder output [B, 512, T]
                teacher_encoder_out:  teacher encoder(clean) output
        """
        # --- Frozen encoder（no_grad 已在 base_model 設定）---
        with torch.no_grad():
            teacher_out, _ = self.base_model.teacher_extractor(clean_audio)
            encoder_out, _ = self.base_model.student_extractor(noisy_audio)
            # 凍結 encoder 的 baseline 重建（用於監控 BWE 帶來的提升）
            recon_before = self.base_model.decode_continuous(encoder_out)

        # --- Trainable LatentBWE ---
        enhanced_latent = self.bwe(encoder_out)   # 梯度從這裡流

        # --- Frozen decoder（需要 enhanced_latent 的梯度通過）---
        # decode_continuous 繞過 @inference_mode，讓梯度可以從 wav 流回 enhanced_latent
        # bandwidth_id 使用 scalar [0]（FiLM norm 廣播 [1,C] 到 [B,T,C]）
        bandwidth_id = torch.tensor([0], device=clean_audio.device)
        x = self.base_model.teacher.backbone(enhanced_latent, bandwidth_id=bandwidth_id)
        recon_wav = self.base_model.teacher.head(x)
        if recon_wav.dim() == 1:
            recon_wav = recon_wav.unsqueeze(0).unsqueeze(0)
        elif recon_wav.dim() == 2:
            recon_wav = recon_wav.unsqueeze(1)

        return {
            'recon_wav':             recon_wav,           # BWE 後（訓練目標）
            'recon_wav_before_bwe':  recon_before,        # BWE 前（baseline）
            'enhanced_latent':       enhanced_latent,
            'encoder_out':           encoder_out,
            'teacher_encoder_out':   teacher_out,
        }

    def get_discrete_tokens(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """推論：取出 enhanced_latent 通過 VQ 的離散 token。

        用於論文架構：enhanced latent → VQ → token → downstream LLM

        Args:
            noisy_audio: [B, 1, T]
        Returns:
            codes: [1, B, 1, T_frame] discrete token indices
        """
        with torch.no_grad():
            encoder_out, _ = self.base_model.student_extractor(noisy_audio)
        enhanced = self.bwe(encoder_out)
        vq_out = self.base_model.vq(enhanced)
        return vq_out['codes']


# ============================================================
# Logging
# ============================================================

class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return False


def setup_logging(output_dir: Path) -> Path:
    import sys as _sys
    log_path = output_dir / 'train.log'
    log_f = open(log_path, 'w', encoding='utf-8', buffering=1)
    _sys.stdout = _TeeIO(_sys.__stdout__, log_f)
    _sys.stderr = _TeeIO(_sys.__stderr__, log_f)
    atexit.register(lambda: log_f.close())
    return log_path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_preinit(device: torch.device, retries: int = 10, sleep_s: float = 2.0):
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
# Data loaders
# ============================================================

def make_val_loader(val_cache_path, batch_size=4, num_workers=2):
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
                mr_stft_fn, mel_fn, scaler=None) -> Dict:
    pipeline.train()
    # encoder 和 decoder 在 base_model 內，已 requires_grad=False，
    # 但 .train() 會影響 BN/Dropout，所以要手動把 frozen 部分設回 eval
    pipeline.base_model.eval()

    metrics = {k: 0.0 for k in [
        'total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
        'mel_loss', 'mrd_fm_loss', 'nan_batches',
    ]}
    n_batches, nan_count = 0, 0

    lw   = config['lambda_wav']
    ls   = config['lambda_stft']
    lm   = config['lambda_mel']
    lfm  = config['lambda_fm']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [LatentBWE]")

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

            loss = (lw * wav_mse + ls * (sc + mag) + lm * mel_loss + lfm * mrd_fm)
            loss = loss / config['grad_accum']

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            metrics['nan_batches'] = nan_count
            optimizer.zero_grad()
            if nan_count >= 10:
                print(f"  Too many NaN batches, aborting epoch!")
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
        n_batches += 1

        pbar.set_postfix({
            'total':  f"{lv:.4f}",
            'wav':    f"{wav_mse.item():.5f}",
            'mrd_fm': f"{mrd_fm.item():.4f}",
        })

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
                  'mel_loss', 'mrd_fm_loss']:
            metrics[k] /= n_batches
    return metrics


@torch.no_grad()
def evaluate(pipeline, mrd_fm_fn, dataloader, device, config,
             mr_stft_fn, mel_fn, max_batches=30) -> Dict:
    pipeline.eval()
    mse_list, noisy_mse_list = [], []
    sc_list, mag_list, mel_list, fm_list = [], [], [], []
    mse_before_list = []

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
        clean_t  = clean[..., :T]
        recon_t  = recon[..., :T]
        noisy_t  = noisy[..., :T]
        Tb = min(clean.shape[-1], before.shape[-1])
        before_t = before[..., :Tb]

        mse_list.append(F.mse_loss(recon_t, clean_t).item())
        mse_before_list.append(F.mse_loss(before_t, clean[..., :Tb]).item())
        noisy_mse_list.append(F.mse_loss(noisy_t, clean_t).item())

        sc, mag = mr_stft_fn(recon_t, clean_t)
        sc_list.append(sc.item()); mag_list.append(mag.item())
        mel_list.append(mel_fn(recon_t, clean_t).item())
        fm_list.append(mrd_fm_fn(recon_t, clean_t).item())

    pipeline.train()
    pipeline.base_model.eval()

    def _m(lst): return float(np.mean(lst)) if lst else float('nan')
    return {
        'val_wav_mse':        _m(mse_list),
        'val_wav_mse_before': _m(mse_before_list),   # BWE 前 baseline
        'val_noisy_mse':      _m(noisy_mse_list),
        'val_stft_sc':        _m(sc_list),
        'val_stft_mag':       _m(mag_list),
        'val_mel_loss':       _m(mel_list),
        'val_mrd_fm':         _m(fm_list),
    }


def _save_audio_samples(pipeline, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    audio_dir = output_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)
    pipeline.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            if saved >= num_samples: break
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


def plot_training_curves(history, output_dir, epoch):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'exp_0229b: LatentBWE (Option 1) — Epoch {epoch}', fontsize=14)
    epochs = range(1, len(history['train_total_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train total', alpha=0.8)
    ax.set_title('Total Loss (train)'); ax.set_yscale('log')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    if history.get('val_noisy_mse'):
        ax.plot(epochs, history['val_noisy_mse'], 'gray', ls=':', label='Noisy baseline')
    if history.get('val_wav_mse_before'):
        ax.plot(epochs, history['val_wav_mse_before'], 'orange', ls='--', label='Before BWE (0227)')
    if history.get('val_wav_mse'):
        ax.plot(epochs, history['val_wav_mse'], 'r-', label='After BWE')
    ax.set_title('Wav MSE (val): before vs after BWE'); ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    if history.get('train_mrd_fm'):
        ax.plot(epochs, history['train_mrd_fm'], 'orange', label='Train MRD FM', alpha=0.8)
    if history.get('val_mrd_fm'):
        ax.plot(epochs, history['val_mrd_fm'], 'red', ls='--', label='Val MRD FM')
    ax.set_title('MRD Feature Matching Loss'); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    if history.get('val_stft_sc'):
        ax.plot(epochs, history['val_stft_sc'], 'b-', label='STFT SC')
    if history.get('val_mel_loss'):
        ax.plot(epochs, history['val_mel_loss'], 'r--', label='Mel')
    ax.set_title('Spectral Losses (val)'); ax.legend(); ax.grid(True)

    ax = axes[2, 0]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'g-', label='LR')
    ax.set_title('Learning Rate'); ax.set_yscale('log'); ax.legend(); ax.grid(True)

    # BWE improvement（val_mse before vs after）
    ax = axes[2, 1]
    if history.get('val_wav_mse') and history.get('val_wav_mse_before'):
        before_arr = np.array(history['val_wav_mse_before'])
        after_arr  = np.array(history['val_wav_mse'])
        improve_pct = (before_arr - after_arr) / (before_arr + 1e-10) * 100
        ax.plot(epochs, improve_pct, 'purple', label='BWE MSE improve %')
        ax.axhline(0, color='black', ls='--', linewidth=0.8)
    ax.set_title('BWE Improvement over 0227 baseline (%)'); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch{epoch:03d}.png', dpi=100)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='exp_0229b: Latent-domain BWE (Option 1)'
    )
    parser.add_argument('--mode', type=str, default='smoke', choices=['smoke', 'epoch'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--encoder_ckpt', type=str, default=EXP0227_BEST_CKPT,
                        help='exp_0227 best_model_val_total.pt (frozen encoder LoRA)')

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

    # LatentBWE 超參數
    parser.add_argument('--bwe_hidden_dim', type=int, default=128,
                        help='BWE 內部隱藏維度（128 → ~0.5M params）')
    parser.add_argument('--bwe_num_blocks', type=int, default=6,
                        help='LatentBWE ResBlock 數量')
    parser.add_argument('--bwe_kernel_size', type=int, default=3)

    parser.add_argument('--lambda_wav',  type=float, default=1.0)
    parser.add_argument('--lambda_stft', type=float, default=1.0)
    parser.add_argument('--lambda_mel',  type=float, default=45.0)
    parser.add_argument('--lambda_fm',   type=float, default=2.0)

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
    BASE_DIR = Path(__file__).parent  # families/eval/bwe_latent/
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    elif args.mode == 'smoke':
        exp_dir = BASE_DIR / f'runs/bwe_latent_smoke_{timestamp}'
    else:
        exp_dir = BASE_DIR / f'runs/bwe_latent_epoch_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    fft_sizes = [int(x) for x in args.stft_fft_sizes.split(',')]
    hop_sizes = [int(x) for x in args.stft_hop_sizes.split(',')]
    win_sizes = [int(x) for x in args.stft_win_sizes.split(',')]

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0229b_latent_bwe'
    config['encoder_source'] = f'exp_0227 best_model_val_total.pt (frozen): {args.encoder_ckpt}'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0229b: Latent-domain BWE — Option 1")
    print("=" * 70)
    print(f"Mode: {args.mode} | Epochs: {args.epochs} | Device: {args.device}")
    print(f"LR: {args.learning_rate} → {args.min_lr}")
    print(f"Encoder LoRA: FROZEN (from exp_0227 ep161 best_model_val_total.pt)")
    print(f"Decoder: FROZEN | MRD: FROZEN (pretrained)")
    print(f"LatentBWE: hidden={args.bwe_hidden_dim}, blocks={args.bwe_num_blocks}, "
          f"kernel={args.bwe_kernel_size}")
    print(f"Loss: λ_wav={args.lambda_wav} + λ_stft={args.lambda_stft} "
          f"+ λ_mel={args.lambda_mel} + λ_fm={args.lambda_fm} (MRD FM)")
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
    print("\nBuilding LatentBWE Pipeline...")
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
                ckpt['model_state_dict'], strict=False
            )
            ep = ckpt.get('epoch', '?')
            print(f"  Loaded exp_0227 model_state_dict (ep{ep})")
            if missing:    print(f"  Missing: {len(missing)}")
            if unexpected: print(f"  Unexpected: {len(unexpected)}")
        # 再次確認凍結（load_state_dict 不改 requires_grad）
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

    mr_stft_fn = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_sizes).to(device)
    mel_fn     = MelReconstructionLoss(SAMPLE_RATE, 1024, 256, 100).to(device)

    # Optimizer：只優化 LatentBWE
    bwe_params = list(pipeline.bwe.parameters())
    trainable_count = sum(p.numel() for p in bwe_params)
    total_count = sum(p.numel() for p in pipeline.parameters())
    print(f"\nTrainable (LatentBWE only): {trainable_count:,} / {total_count:,} "
          f"({100*trainable_count/total_count:.3f}%)")

    optimizer = torch.optim.AdamW(
        bwe_params, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    def lr_lambda(ep):
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
        'train_mrd_fm': [],
        'val_wav_mse': [], 'val_wav_mse_before': [], 'val_noisy_mse': [],
        'val_stft_sc': [], 'val_stft_mag': [], 'val_mel_loss': [],
        'val_mrd_fm': [], 'lr': [],
    }

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_epoch(
            pipeline, mrd_fm_fn, train_loader, optimizer, device, epoch, config,
            mr_stft_fn, mel_fn, scaler,
        )
        val_m = evaluate(
            pipeline, mrd_fm_fn, val_loader, device, config,
            mr_stft_fn, mel_fn, args.eval_max_batches,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_total_loss'].append(train_m['total_loss'])
        history['train_wav_mse'].append(train_m['wav_mse'])
        history['train_stft_sc'].append(train_m['stft_sc_loss'])
        history['train_stft_mag'].append(train_m['stft_mag_loss'])
        history['train_mel'].append(train_m['mel_loss'])
        history['train_mrd_fm'].append(train_m['mrd_fm_loss'])
        history['val_wav_mse'].append(val_m['val_wav_mse'])
        history['val_wav_mse_before'].append(val_m['val_wav_mse_before'])
        history['val_noisy_mse'].append(val_m['val_noisy_mse'])
        history['val_stft_sc'].append(val_m['val_stft_sc'])
        history['val_stft_mag'].append(val_m['val_stft_mag'])
        history['val_mel_loss'].append(val_m['val_mel_loss'])
        history['val_mrd_fm'].append(val_m['val_mrd_fm'])
        history['lr'].append(lr_now)

        val_mse   = val_m['val_wav_mse']
        noisy_mse = val_m['val_noisy_mse']
        before_mse = val_m['val_wav_mse_before']
        val_total = (val_mse
                     + val_m['val_stft_sc']
                     + val_m['val_stft_mag']
                     + args.lambda_mel * val_m['val_mel_loss'])
        improve_vs_noisy = (noisy_mse - val_mse) / noisy_mse * 100 if noisy_mse > 0 else 0
        improve_vs_before = (before_mse - val_mse) / before_mse * 100 if before_mse > 0 else 0

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: total={train_m['total_loss']:.4f}  "
              f"wav={train_m['wav_mse']:.5f}  "
              f"mrd_fm={train_m['mrd_fm_loss']:.4f}")
        print(f"  Val:   mse_after={val_mse:.5f}  mse_before={before_mse:.5f}  "
              f"(BWE Δ={improve_vs_before:+.1f}%)  "
              f"noisy={noisy_mse:.5f}  total={val_total:.4f}")
        print(f"         stft_sc={val_m['val_stft_sc']:.4f}  "
              f"mel={val_m['val_mel_loss']:.4f}  "
              f"mrd_fm={val_m['val_mrd_fm']:.4f}")
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
            print(f"  ★ New best val_total: {best_val_total:.4f} → saved best_model_val_total.pt")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(save_dict, exp_dir / 'best_model.pt')
            print(f"  ★ New best val_mse: {best_val_mse:.5f} → saved best_model.pt")

        if epoch % args.save_checkpoint_every == 0:
            torch.save(save_dict, exp_dir / f'checkpoint_epoch{epoch:03d}.pt')

        if epoch % args.save_audio_interval == 0 or epoch == args.epochs:
            plot_training_curves(history, exp_dir, epoch)
            _save_audio_samples(pipeline, val_loader, device, exp_dir, epoch,
                                num_samples=2, split='val')
            _save_audio_samples(pipeline, train_loader, device, exp_dir, epoch,
                                num_samples=2, split='train')

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_total: {best_val_total:.4f}")
    print(f"Output: {exp_dir}")


if __name__ == '__main__':
    main()

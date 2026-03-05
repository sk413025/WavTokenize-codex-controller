"""
exp_0305: 選擇性 LoRA 層實驗訓練腳本

科學問題：
  exp_0304 feature map 分析指出 18 層中只有少數層是「噪音敏感」或「高頻重要」。
  全層 LoRA (exp_0224a) 是否浪費了參數預算在對去噪沒有幫助的層上？
  選擇性 LoRA 能否用更少的參數達到更好（或相當）的去噪效果？

三種 Plan 比較：
  plan_a — adapt_top6 (6 層, rank=32, ~245K trainable)
  plan_b — adapt_top8 (8 層, rank=32, ~307K trainable)   ★ 推薦首選
  plan_c — all_18 equivalent-budget (18 層, rank=10, ~288K)

Baseline 參照：
  exp_0224a: all_18 rank=64 (926K trainable), val_wav_mse best=0.0233

執行方式：
  # Smoke test
  python exp_0305/train_selective_lora.py --plan plan_b --mode smoke

  # 正式訓練 plan_a
  python exp_0305/train_selective_lora.py --plan plan_a --epochs 300 --device cuda:0

  # 正式訓練 plan_b (推薦)
  python exp_0305/train_selective_lora.py --plan plan_b --epochs 300 --device cuda:0

  # 正式訓練 plan_c (等參數對照)
  python exp_0305/train_selective_lora.py --plan plan_c --epochs 300 --device cuda:0

Loss（與 exp_0224a 完全相同，確保公平比較）：
  λ_wav=1.0  * MSE(recon_wav, clean_wav)
  λ_stft=1.0 * MR-STFT(recon_wav, clean_wav)
  λ_mel=45.0 * Mel(recon_wav, clean_wav)

初始化：exp_0217 best checkpoint (exp_0224a 同款起點)
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0305.models_selective_lora import TeacherStudentSelectiveLoRA, PLANS
from exp_0216.data_augmented import AugmentedCurriculumDataset, collate_fn_curriculum
from encoder.modules.conv import SConv1d
from encoder.modules.seanet import SEANetResnetBlock

# ── exp_0217 best ckpt (與 exp_0224a 相同起點) ─────────────────────────────
EXP0217_BEST_CKPT = (
    Path(__file__).parent.parent /
    'exp_0217/runs/t453_weighted_epoch_20260217_104843/best_model.pt'
)

SAMPLE_RATE = 24000


# ============================================================
# Anchor Regularization 基礎設施（用於 plan_b 穩定層約束）
# ============================================================

def build_conv18_modules(encoder: torch.nn.Module) -> dict:
    """將 SEANet encoder 拆解為按纊號排列的16層 conv 模組的字典。

    Args:
        encoder: SEANet encoder 物件（.model 屬性為所有操作層的 list）。

    Returns:
        Dict[int, SConv1d]，鍵為 layer id 0..17。
    """
    modules: dict = {}
    li = 0
    for m in encoder.model:
        if isinstance(m, SConv1d):
            modules[li] = m; li += 1
        elif isinstance(m, SEANetResnetBlock):
            modules[li] = m.block[1]; li += 1
            modules[li] = m.block[3]; li += 1
            if isinstance(m.shortcut, SConv1d):
                modules[li] = m.shortcut; li += 1
    if len(modules) != 18:
        raise RuntimeError(f"Expected 18 modules, got {len(modules)}")
    return modules


class LayerHookBank:
    """對指定層的 forward hook，用於擷取中間特徵供 anchor loss 計算。

    Args:
        layer_modules: build_conv18_modules 回傳的 dict。
        layer_ids: 要錨定的層編號列表。
    """
    def __init__(self, layer_modules: dict, layer_ids: list):
        self.layer_ids = layer_ids
        self.cache: dict = {}
        self.handles = []
        for li in layer_ids:
            self.handles.append(layer_modules[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, li: int):
        """建立特定層索引的 hook 函式。

        Args:
            li: 層索引。

        Returns:
            hook 函式。
        """
        def _hook(_m, _inp, out):
            self.cache[li] = out[0] if isinstance(out, tuple) else out
        return _hook

    def clear(self):
        """.清空已缓存的特徵。"""
        self.cache.clear()

    def close(self):
        """移除所有 hook。"""
        for h in self.handles: h.remove()
        self.handles = []


def compute_anchor_loss(student_cache: dict, anchor_cache: dict, layer_ids: list) -> torch.Tensor:
    """計算層略性錨定 loss：將 student 中間層特徵拉向原始 WavTokenizer。

    Args:
        student_cache: 由 student hooks 擷取的特徵 dict。
        anchor_cache: 由 teacher (anchor) hooks 擷取的特徵 dict。
        layer_ids: 參與計算的層編號列表。

    Returns:
        各層 MSE 的平均値（pure torch.Tensor）。
    """
    total = sum(F.mse_loss(student_cache[li], anchor_cache[li].detach()) for li in layer_ids if li in student_cache and li in anchor_cache)
    n = sum(1 for li in layer_ids if li in student_cache and li in anchor_cache)
    device = next(iter(student_cache.values())).device if student_cache else torch.device('cpu')
    return total / max(n, 1) if n > 0 else torch.tensor(0.0, device=device)


# ============================================================
# Multi-Resolution STFT Loss（與 exp_0224a 完全相同）
# ============================================================

class STFTLoss(nn.Module):
    """單解析度 STFT 損失。

    Args:
        n_fft: FFT 點數
        hop_length: hop 長度
        win_length: 視窗長度
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        """初始化 STFTLoss。

        Args:
            n_fft: FFT 點數
            hop_length: hop 長度
            win_length: 視窗長度
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """計算 STFT 幅度。

        Args:
            x: [B, T] 波形

        Returns:
            [B, F, T'] 幅度頻譜圖
        """
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        )
        return torch.abs(spec)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向傳播，計算 spectral convergence + log magnitude loss。

        Args:
            y_hat: 重建音頻 [B, T] 或 [B, 1, T]
            y: 目標音頻 [B, T] 或 [B, 1, T]

        Returns:
            (sc_loss, log_mag_loss)
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
    """多解析度 STFT 損失，用於捕捉不同頻率細節。

    Args:
        fft_sizes: FFT 尺寸列表
        hop_sizes: hop 尺寸列表
        win_sizes: 視窗尺寸列表
    """

    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512],
        hop_sizes: List[int] = [512, 256, 128],
        win_sizes: List[int] = [2048, 1024, 512],
    ):
        """初始化 MultiResolutionSTFTLoss。

        Args:
            fft_sizes: FFT 尺寸列表（由粗到細）
            hop_sizes: hop 尺寸列表
            win_sizes: 視窗尺寸列表
        """
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft, hop, win)
            for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算多解析度平均 STFT loss。

        Args:
            y_hat: 重建音頻
            y: 目標音頻

        Returns:
            (sc_loss, mag_loss) 多解析度平均值
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
    """Mel 頻譜重建損失（log-mel L1）。

    Args:
        sample_rate: 取樣率
        n_fft: FFT 點數
        hop_length: hop 長度
        n_mels: Mel 濾波器數量
    """

    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        """初始化 MelReconstructionLoss。

        Args:
            sample_rate: 取樣率（Hz）
            n_fft: FFT 點數
            hop_length: hop 長度
            n_mels: Mel 濾波器數量
        """
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, center=True, power=1,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """計算 log-Mel L1 損失。

        Args:
            y_hat: 重建音頻
            y: 目標音頻

        Returns:
            Mel L1 損失
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
    """同時寫入多個 stream 的 IO wrapper。"""

    def __init__(self, *streams):
        """初始化 _TeeIO。

        Args:
            *streams: 要同時寫入的 stream 列表
        """
        self._streams = streams

    def write(self, data):
        """寫入所有 streams。

        Args:
            data: 要寫入的資料
        """
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        """刷新所有 streams。"""
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        """返回 False，表示不是 tty。

        Returns:
            False
        """
        return False


def setup_logging(output_dir: Path) -> Path:
    """設置 stdout/stderr 同步寫入 log 檔。

    Args:
        output_dir: 輸出目錄路徑

    Returns:
        log 檔路徑
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
    """固定所有隨機種子以確保可重現性。

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
    """預先初始化 CUDA 裝置，避免啟動時 OOM。

    Args:
        device: CUDA 裝置
        retries: 重試次數
        sleep_s: 每次重試等待秒數
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


def make_loaders(batch_size: int = 8, num_workers: int = 2, config: dict = None):
    """建立 train/val DataLoader。

    Args:
        batch_size: 批次大小
        num_workers: 資料讀取執行緒數
        config: 訓練設定字典

    Returns:
        (train_loader, val_loader)
    """
    cfg = config or {}
    train_ds = AugmentedCurriculumDataset(
        TRAIN_CACHE, augment=True,
        filter_clean_to_clean=True, compute_snr=False,
        snr_remix_prob=cfg.get('snr_remix_prob', 0.5),
        snr_remix_range=(cfg.get('snr_remix_min', -5.0), cfg.get('snr_remix_max', 25.0)),
        random_gain_prob=cfg.get('random_gain_prob', 0.3),
        random_gain_db=cfg.get('random_gain_db', 3.0),
        random_crop_prob=cfg.get('random_crop_prob', 0.3),
        random_crop_min_ratio=cfg.get('random_crop_min_ratio', 0.7),
        time_stretch_prob=cfg.get('time_stretch_prob', 0.2),
        time_stretch_range=(cfg.get('time_stretch_min', 0.95), cfg.get('time_stretch_max', 1.05)),
    )
    val_ds = AugmentedCurriculumDataset(
        VAL_CACHE, augment=False,
        filter_clean_to_clean=True, compute_snr=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_curriculum,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_curriculum,
    )
    return train_loader, val_loader


# ============================================================
# Train / Eval
# ============================================================

def train_epoch(
    model: TeacherStudentSelectiveLoRA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    mr_stft_fn: MultiResolutionSTFTLoss,
    mel_fn: MelReconstructionLoss,
    scaler: GradScaler = None,
    anchor_encoder=None,
    student_hooks: LayerHookBank = None,
    anchor_hooks: LayerHookBank = None,
) -> Dict:
    """執行一個訓練 epoch。

    損失函數與 exp_0224a 一致：
        λ_wav * MSE + λ_stft * MR-STFT + λ_mel * Mel
        [+ λ_anchor * Anchor]（如果 plan 有 anchor_layer_ids）

    Args:
        model: TeacherStudentSelectiveLoRA 模型
        dataloader: 訓練資料載入器
        optimizer: 優化器
        device: 計算裝置
        epoch: 當前 epoch 編號
        config: 訓練設定字典
        mr_stft_fn: 多解析度 STFT 損失函數
        mel_fn: Mel 損失函數
        scaler: AMP GradScaler（可選）
        anchor_encoder: frozen teacher encoder，不為 None 時啟用 anchor loss
        student_hooks: LayerHookBank 掻取 student 中間層
        anchor_hooks: LayerHookBank 掻取 anchor 中間層

    Returns:
        包含 total_loss, wav_mse, stft_sc, stft_mag, mel_loss, anchor_loss 的 metrics dict
    """
    model.train()
    # Decoder（teacher backbone+head）保持 eval
    model.teacher.backbone.eval()
    model.teacher.head.eval()

    λ_wav = config['lambda_wav']
    λ_stft = config['lambda_stft']
    λ_mel = config['lambda_mel']
    λ_anchor = config.get('lambda_anchor', 0.0)
    anchor_layer_ids = config.get('anchor_layer_ids', [])
    accum = config['grad_accum']

    metrics = dict(total_loss=0, wav_mse=0, stft_sc=0, stft_mag=0, mel_loss=0, anchor_loss=0, nan_batches=0)
    n_batches = 0
    nan_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{model.plan}]")
    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)

        if batch_idx % accum == 0:
            optimizer.zero_grad()

        # 清除 hook cache（只有啟用 anchor 時才需要）
        if anchor_encoder is not None:
            if student_hooks: student_hooks.clear()
            if anchor_hooks: anchor_hooks.clear()
            with torch.no_grad():
                _ = anchor_encoder(noisy)

        with autocast(enabled=config['use_amp']):
            out = model.forward_wav(clean, noisy)
            recon = out['recon_wav']
            T = min(clean.shape[-1], recon.shape[-1])
            recon_t, clean_t = recon[..., :T], clean[..., :T]

            wav_mse = F.mse_loss(recon_t, clean_t)
            sc, mag = mr_stft_fn(recon_t, clean_t)
            mel = mel_fn(recon_t, clean_t)
            # Anchor loss（只有 plan_b 且已設置 hooks 時有效）
            anchor = (
                compute_anchor_loss(student_hooks.cache, anchor_hooks.cache, anchor_layer_ids)
                if anchor_encoder is not None and student_hooks and anchor_hooks
                else torch.tensor(0.0, device=device)
            )
            loss = (λ_wav * wav_mse + λ_stft * (sc + mag) + λ_mel * mel + λ_anchor * anchor) / accum

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            metrics['nan_batches'] = nan_count
            optimizer.zero_grad()
            if nan_count >= 10:
                print(f"  ⚠ Too many NaN ({nan_count}), aborting epoch!")
                break
            continue

        if scaler:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config['grad_clip'],
                )
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (batch_idx + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config['grad_clip'],
                )
                optimizer.step()

        lv = loss.item() * accum
        metrics['total_loss'] += lv
        metrics['wav_mse'] += wav_mse.item()
        metrics['stft_sc'] += sc.item()
        metrics['stft_mag'] += mag.item()
        metrics['mel_loss'] += mel.item()
        metrics['anchor_loss'] += anchor.item()
        n_batches += 1
        pbar.set_postfix(total=f"{lv:.3f}", wav=f"{wav_mse.item():.5f}", anchor=f"{anchor.item():.5f}")

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc', 'stft_mag', 'mel_loss', 'anchor_loss']:
            metrics[k] /= n_batches
    return metrics


@torch.no_grad()
def evaluate(
    model: TeacherStudentSelectiveLoRA,
    dataloader: DataLoader,
    device: torch.device,
    mr_stft_fn: MultiResolutionSTFTLoss,
    mel_fn: MelReconstructionLoss,
    max_batches: int = 30,
    anchor_encoder=None,
    student_hooks: LayerHookBank = None,
    anchor_hooks: LayerHookBank = None,
    anchor_layer_ids: list = None,
    lambda_anchor: float = 0.0,
) -> Dict:
    """評估模型在 validation 集上的表現。

    Args:
        model: TeacherStudentSelectiveLoRA 模型
        dataloader: 驗證資料載入器
        device: 計算裝置
        mr_stft_fn: STFT 損失
        mel_fn: Mel 損失
        max_batches: 最多評估批次
        anchor_encoder: frozen teacher encoder
        student_hooks: student 中間層 hooks
        anchor_hooks: anchor 中間層 hooks
        anchor_layer_ids: 錨定層編號列表
        lambda_anchor: anchor loss 權重

    Returns:
        包含各項指標的 dict
    """
    model.eval()
    wav_list, noisy_list, sc_list, mel_list, noisy_sc_list, anchor_list = [], [], [], [], [], []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)

        if anchor_encoder is not None:
            if student_hooks: student_hooks.clear()
            if anchor_hooks: anchor_hooks.clear()
            _ = anchor_encoder(noisy)

        out = model.forward_wav(clean, noisy)
        recon = out['recon_wav']
        T = min(clean.shape[-1], recon.shape[-1], noisy.shape[-1])
        c, r, n = clean[..., :T], recon[..., :T], noisy[..., :T]

        wav_list.append(F.mse_loss(r, c).item())
        sc, _ = mr_stft_fn(r, c)
        sc_list.append(sc.item())
        mel_list.append(mel_fn(r, c).item())
        noisy_list.append(F.mse_loss(n, c).item())
        sc_n, _ = mr_stft_fn(n, c)
        noisy_sc_list.append(sc_n.item())
        if anchor_encoder is not None and student_hooks and anchor_hooks and anchor_layer_ids:
            anc = compute_anchor_loss(student_hooks.cache, anchor_hooks.cache, anchor_layer_ids)
            anchor_list.append(anc.item())

    model.train()
    _m = lambda lst: float(np.mean(lst)) if lst else float('nan')
    return {
        'val_wav_mse': _m(wav_list),
        'val_noisy_mse': _m(noisy_list),
        'val_stft_sc': _m(sc_list),
        'val_mel_loss': _m(mel_list),
        'val_noisy_stft_sc': _m(noisy_sc_list),
        'val_anchor': _m(anchor_list) if anchor_list else float('nan'),
    }


# ============================================================
# Plotting
# ============================================================

def plot_curves(history: dict, output_dir: Path, epoch: int, plan: str):
    """繪製訓練曲線並儲存圖片。

    Args:
        history: 訓練歷史字典
        output_dir: 輸出目錄
        epoch: 當前 epoch 編號
        plan: 實驗 plan 名稱（用於標題）
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'exp_0305 / {plan}  (Epoch {epoch})', fontsize=13)
    ep = range(1, len(history.get('train_total_loss', [])) + 1)

    def _plot(ax, keys, title, log=False):
        """在指定 ax 上繪製指定 keys 的曲線。

        Args:
            ax: matplotlib Axes 物件
            keys: 要繪製的 key 列表（每個元素為 (key, label, color)）
            title: 子圖標題
            log: 是否使用對數 y 軸
        """
        for key, label, color in keys:
            if history.get(key):
                ax.plot(ep, history[key][:len(ep)], color=color, label=label, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        if log:
            ax.set_yscale('log')

    _plot(axes[0, 0], [('train_total_loss', 'Train total', 'blue')], 'Total Loss (train)', log=True)
    _plot(axes[0, 1],
          [('val_noisy_mse', 'Noisy baseline', 'gray'),
           ('val_wav_mse', 'Recon MSE', 'red')],
          'Wav MSE (val)')
    _plot(axes[0, 2],
          [('train_stft_sc', 'STFT SC', 'cyan'),
           ('train_stft_mag', 'STFT Mag', 'magenta')],
          'STFT Loss (train)')
    _plot(axes[1, 0],
          [('train_mel_loss', 'Mel (train)', 'orange'),
           ('val_mel_loss', 'Mel (val)', 'darkorange')],
          'Mel Loss')
    _plot(axes[1, 1],
          [('val_stft_sc', 'STFT SC (val)', 'teal'),
           ('val_noisy_stft_sc', 'Noisy SC (val)', 'silver')],
          'STFT SC (val)')
    # 改善率圖
    ax = axes[1, 2]
    if history.get('val_wav_mse') and history.get('val_noisy_mse'):
        n_pts = min(len(history['val_wav_mse']), len(history['val_noisy_mse']))
        impr = [
            (history['val_noisy_mse'][i] - history['val_wav_mse'][i]) / (history['val_noisy_mse'][i] + 1e-9)
            for i in range(n_pts)
        ]
        ax.plot(range(1, n_pts + 1), impr, 'g-', label='MSE improvement ratio', alpha=0.85)
        ax.axhline(0, color='k', ls='--', alpha=0.3)
        ax.set_title('Denoising Improvement (val)')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    date_str = datetime.now().strftime('%Y%m%d')
    fname = output_dir / f'training_curves_exp0305_{plan}_epoch{epoch:03d}_{date_str}_plot_curves.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  圖表儲存：{fname.name}")


@torch.no_grad()
def save_audio_samples(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    n_samples: int = 4,
    sample_rate: int = 24000,
):
    """存儲 val 音檔樣本：noisy / clean / recon 各一份以便聽感評估。

    Args:
        model: 當前模型。
        val_loader: 驗證 DataLoader。
        device: 運算裝置。
        output_dir: 輸出目錄。
        epoch: 當前 epoch 編號（用於檔名）。
        n_samples: 存儲筆數。
        sample_rate: 音訊取樣率。
    """
    audio_dir = output_dir / 'audio_samples' / f'epoch{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in val_loader:
        noisy = batch['noisy_audio'].to(device)
        clean = batch['clean_audio'].to(device)
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)
        out = model.forward_wav(clean, noisy)
        recon = out['recon_wav']
        for i in range(min(noisy.shape[0], n_samples - saved)):
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_noisy.wav'), noisy[i].cpu(), sample_rate)
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_clean.wav'), clean[i].cpu(), sample_rate)
            r = recon[i].cpu()
            if r.dim() == 1:
                r = r.unsqueeze(0)
            torchaudio.save(str(audio_dir / f'sample{saved+i:02d}_recon.wav'), r, sample_rate)
        saved += min(noisy.shape[0], n_samples - saved)
        if saved >= n_samples:
            break
    print(f'  [音檔] 已存儲 {saved} 筆 val 音檔至 {audio_dir}')


# ============================================================
# Main
# ============================================================

def parse_args():
    """解析命令列參數。

    Returns:
        argparse.Namespace 物件
    """
    p = argparse.ArgumentParser(description='exp_0305 Selective LoRA')
    p.add_argument('--plan', type=str, default='plan_b',
                   choices=['plan_a', 'plan_b', 'plan_c'],
                   help='LoRA 層選擇方案 (see models_selective_lora.py)')
    p.add_argument('--mode', type=str, default='epoch',
                   choices=['smoke', 'epoch'])
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--grad_accum', type=int, default=2)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--lambda_wav', type=float, default=1.0)
    p.add_argument('--lambda_stft', type=float, default=1.0)
    p.add_argument('--lambda_mel', type=float, default=45.0)
    p.add_argument('--lambda_anchor', type=float, default=None,
                   help='Anchor loss 權重（None = 使用 PLANS 中設定的預設値）')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=str, default=None,
                   help='從 checkpoint 繼續訓練的路徑')
    return p.parse_args()


def main():
    """exp_0305 主訓練流程。"""
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    cuda_preinit(device)

    is_smoke = args.mode == 'smoke'
    epochs = 3 if is_smoke else args.epochs

    # ── 輸出目錄 ────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(__file__).parent / 'runs' / f'{args.plan}_epoch_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    print(f"\n{'='*65}")
    print(f"  exp_0305 Selective LoRA — {args.plan}")
    print(f"  output: {out_dir}")
    print(f"  epochs: {epochs}  device: {device}")
    print(f"{'='*65}\n")

    # ── 設定 ─────────────────────────────────────────────────────
    config = {
        'plan': args.plan,
        'mode': args.mode,
        'epochs': epochs,
        'batch_size': args.batch_size if not is_smoke else 4,
        'grad_accum': args.grad_accum,
        'learning_rate': args.lr,
        'min_lr': args.min_lr,
        'warmup_epochs': args.warmup,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'use_amp': not args.no_amp,
        'lora_dropout': args.lora_dropout,
        'lambda_wav': args.lambda_wav,
        'lambda_stft': args.lambda_stft,
        'lambda_mel': args.lambda_mel,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'timestamp': ts,
        'experiment': f'exp_0305_selective_lora_{args.plan}',
        # 資料增強（與 exp_0224a 相同）
        'snr_remix_prob': 0.5, 'snr_remix_min': -5.0, 'snr_remix_max': 25.0,
        'random_gain_prob': 0.3, 'random_gain_db': 3.0,
        'random_crop_prob': 0.3, 'random_crop_min_ratio': 0.7,
        'time_stretch_prob': 0.2, 'time_stretch_min': 0.95, 'time_stretch_max': 1.05,
        # plan_info
        'plan_n_layers': PLANS[args.plan]['n_layers'],
        'plan_rank': PLANS[args.plan]['rank'],
        'plan_alpha': PLANS[args.plan]['alpha'],
        'plan_description': PLANS[args.plan]['description'],
        # anchor 設定（只有 plan_b 有 anchor_layer_ids）
        'anchor_layer_ids': PLANS[args.plan].get('anchor_layer_ids', []),
        'lambda_anchor': (
            args.lambda_anchor
            if args.lambda_anchor is not None
            else PLANS[args.plan].get('lambda_anchor', 0.0)
        ),
        # baseline reference
        'baseline_exp': 'exp_0224a',
        'baseline_val_wav_mse_best': 0.0233,
        'evidence_source': 'exp_0304/wavtokenizer_featuremap_14wav_extended',
    }

    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ── 模型 ─────────────────────────────────────────────────────
    model = TeacherStudentSelectiveLoRA(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        plan=args.plan,
        lora_dropout=args.lora_dropout,
        device=str(device),
    )

    # Student encoder 使用官方 WavTokenizer 預訓練權重（TeacherStudentSelectiveLoRA.__init__ 已完成）
    # 不載入任何 exp_0217/0224a 先驗，確保 base weights 不偏離官方清晰音質空間
    print("Student encoder: 使用官方 WavTokenizer 預訓練權重（NO exp_0217/0224a 先驗）")

    # ── Resume ───────────────────────────────────────────────────
    start_epoch = 0
    history = {k: [] for k in [
        'train_total_loss', 'train_wav_mse', 'train_stft_sc', 'train_stft_mag', 'train_mel_loss',
        'train_anchor_loss',
        'val_wav_mse', 'val_noisy_mse', 'val_stft_sc', 'val_mel_loss', 'val_noisy_stft_sc',
        'val_anchor', 'lr',
    ]}
    best_val_mse = float('inf')
    best_val_total = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        history = ckpt.get('history', history)
        best_val_mse = ckpt.get('best_val_mse', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # ── 損失函數 ─────────────────────────────────────────────────
    mr_stft = MultiResolutionSTFTLoss().to(device)
    mel_loss = MelReconstructionLoss().to(device)

    # ── 優化器 + 排程器 ──────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config['learning_rate'], weight_decay=config['weight_decay'],
    )
    # cosine annealing with warmup
    total_steps = epochs - config['warmup_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=config['min_lr'],
    )
    scaler = GradScaler() if config['use_amp'] else None

    # ── Data loaders ─────────────────────────────────────────────
    n_train = 20 if is_smoke else None
    train_loader, val_loader = make_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        config=config,
    )
    if is_smoke:
        from torch.utils.data import Subset
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(20)), batch_size=4, shuffle=True,
            num_workers=0, collate_fn=collate_fn_curriculum,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(10)), batch_size=4, shuffle=False,
            num_workers=0, collate_fn=collate_fn_curriculum,
        )

    print(f"\nTrain batches/epoch: {len(train_loader)}")
    print(f"Val   batches/eval:  min({len(val_loader)}, 30)")
    print(f"Plan description: {PLANS[args.plan]['description']}\n")

    # ── Anchor 初始化（根據 PLANS 設定自動決定是否啟用）─────────────────────────
    anchor_layer_ids = config['anchor_layer_ids']   # plan_b: [1,4,5,7,10,13,14,15,16,17]; 其他: []
    anchor_encoder = student_hooks = anchor_hooks = None
    if anchor_layer_ids:
        print(f"[Anchor] 啟用 anchor 約束，錨定層 = {anchor_layer_ids}")
        print(f"[Anchor] 錨定目標 = 原始 WavTokenizer teacher encoder（frozen）")
        anchor_encoder = model.teacher.feature_extractor.encodec.encoder.eval()
        for p in anchor_encoder.parameters():
            p.requires_grad_(False)
        student_enc  = model.student.feature_extractor.encodec.encoder
        student_mods = build_conv18_modules(student_enc)
        anchor_mods  = build_conv18_modules(anchor_encoder)
        student_hooks = LayerHookBank(student_mods, anchor_layer_ids)
        anchor_hooks  = LayerHookBank(anchor_mods,  anchor_layer_ids)
    else:
        print(f"[Anchor] plan={args.plan} 無 anchor 約束（純 LoRA 對照組）")

    # ── 訓練迴圈 ─────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, epochs + 1):
        # Warmup LR
        if epoch <= config['warmup_epochs']:
            warmup_factor = epoch / config['warmup_epochs']
            for pg in optimizer.param_groups:
                pg['lr'] = config['learning_rate'] * warmup_factor

        t_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config,
            mr_stft, mel_loss, scaler,
            anchor_encoder=anchor_encoder,
            student_hooks=student_hooks,
            anchor_hooks=anchor_hooks,
        )

        if epoch > config['warmup_epochs']:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]['lr']

        # Eval
        v_metrics = evaluate(
            model, val_loader, device, mr_stft, mel_loss,
            anchor_encoder=anchor_encoder,
            student_hooks=student_hooks,
            anchor_hooks=anchor_hooks,
            anchor_layer_ids=anchor_layer_ids,
            lambda_anchor=config['lambda_anchor'],
        )

        # History
        history['train_total_loss'].append(t_metrics['total_loss'])
        history['train_wav_mse'].append(t_metrics['wav_mse'])
        history['train_stft_sc'].append(t_metrics['stft_sc'])
        history['train_stft_mag'].append(t_metrics['stft_mag'])
        history['train_mel_loss'].append(t_metrics['mel_loss'])
        history['train_anchor_loss'].append(t_metrics.get('anchor_loss', 0.0))
        history['val_wav_mse'].append(v_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(v_metrics['val_noisy_mse'])
        history['val_stft_sc'].append(v_metrics['val_stft_sc'])
        history['val_mel_loss'].append(v_metrics['val_mel_loss'])
        history['val_noisy_stft_sc'].append(v_metrics['val_noisy_stft_sc'])
        history['val_anchor'].append(v_metrics.get('val_anchor', float('nan')))
        history['lr'].append(cur_lr)

        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print(
            f"[Epoch {epoch:3d}]  "
            f"train_total={t_metrics['total_loss']:.4f}  "
            f"val_mse={v_metrics['val_wav_mse']:.5f}  "
            f"(noisy={v_metrics['val_noisy_mse']:.5f})  "
            f"anchor={t_metrics.get('anchor_loss',0.0):.5f}  "
            f"stft_sc={v_metrics['val_stft_sc']:.4f}  "
            f"lr={cur_lr:.2e}"
        )

        # Best val_wav_mse
        if v_metrics['val_wav_mse'] < best_val_mse:
            best_val_mse = v_metrics['val_wav_mse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mse': best_val_mse,
                'history': history,
                'config': config,
            }, out_dir / 'best_model.pt')
            print(f"  ✓ best_model.pt 更新 (val_mse={best_val_mse:.5f})")

        # Checkpoint
        if epoch % 10 == 0 or is_smoke:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mse': best_val_mse,
                'history': history,
                'config': config,
            }, out_dir / f'checkpoint_epoch{epoch:03d}.pt')

        # Plot
        if epoch % 25 == 0 or (is_smoke and epoch == epochs):
            plot_curves(history, out_dir, epoch, args.plan)

        # Audio 樣本：每 10 epoch 或 smoke 模式結束時
        if epoch % 10 == 0 or (is_smoke and epoch == epochs):
            save_audio_samples(model, val_loader, device, out_dir, epoch)

    # ── 訓練結束 ─────────────────────────────────────────────────
    # 關閉 anchor hooks
    if student_hooks: student_hooks.close()
    if anchor_hooks: anchor_hooks.close()
    # 訓練結束後存最終 loss 圖
    plot_curves(history, out_dir, args.epochs if not is_smoke else epoch, args.plan)
    print(f"\n{'='*65}")
    print(f"  exp_0305 / {args.plan} 訓練完成")
    print(f"  Best val_wav_mse: {best_val_mse:.5f}")
    print(f"  Baseline (exp_0224a): 0.02330")
    delta = best_val_mse - 0.02330
    print(f"  vs Baseline: {'+' if delta >= 0 else ''}{delta:.5f} "
          f"({'worse' if delta >= 0 else '★ better'})")
    print(f"  Output: {out_dir}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()

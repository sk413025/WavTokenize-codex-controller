"""
exp_0304 備用方案: 材質對抗訓練 (Domain Adversarial Training, 方法 3)

在 train_material_gen.py (方法 1+2) 的基礎上額外加入：
    MaterialClassifier — 分類 encoder features 屬於哪種材質
    GradientReversalLayer — 讓 encoder 學到材質不變的特徵表示

Loss（在方法 1+2 基礎上新增）：
    + λ_adv * CrossEntropy(material_pred, material_label)   ← GRL 反轉梯度

設計動機：
    如果方法 1+2（頻率增強）不足以讓模型泛化到未知材質，
    可在 feature space 層面強制讓 encoder 忽略材質相關資訊。
    Gradient Reversal 使 encoder 輸出的 features 無法被分類器區分材質，
    從而學到材質不變 (material-invariant) 的語音表示。

執行：
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python exp_0304/train_material_adv.py --mode smoke

    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python exp_0304/train_material_adv.py \\
        --mode epoch --epochs 300 --device cuda:0
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
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torch.autograd import Function

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0224.models_no_vq import TeacherStudentNoVQ
from exp_0304.data_material_aug import MaterialAugDataset, create_material_aug_dataloaders
from exp_0216.data_augmented import collate_fn_curriculum


# ============================================================
# Constants
# ============================================================

SAMPLE_RATE = 24000
MATERIAL_TO_IDX = {'box': 0, 'papercup': 1, 'plastic': 2}
NUM_MATERIALS = len(MATERIAL_TO_IDX)

EXP0227_BEST_CKPT = (
    'exp_0227/runs/enc_mrd_fm_epoch_20260227_024953/best_model_val_total.pt'
)


# ============================================================
# Gradient Reversal Layer
# ============================================================

class GradientReversalFunction(Function):
    """梯度反轉函式，用於 Domain Adversarial Training。

    forward 時直接傳遞，backward 時將梯度乘以 -lambda。
    """

    @staticmethod
    def forward(ctx, x, lambda_val):
        """前向傳播，直接傳遞輸入。

        Args:
            ctx: 上下文物件。
            x: 輸入張量。
            lambda_val: 梯度反轉係數。

        Returns:
            原封不動的輸入張量。
        """
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """反向傳播，反轉梯度。

        Args:
            ctx: 上下文物件。
            grad_output: 上游梯度。

        Returns:
            反轉後的梯度。
        """
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """梯度反轉層，包裝 GradientReversalFunction。

    Args:
        lambda_val: 梯度反轉係數，可動態調整。
    """

    def __init__(self, lambda_val: float = 1.0):
        """初始化 GradientReversalLayer。

        Args:
            lambda_val: 初始梯度反轉係數。
        """
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        """前向傳播，透過 GradientReversalFunction。

        Args:
            x: 輸入張量。

        Returns:
            梯度反轉後的張量（forward 時不變）。
        """
        return GradientReversalFunction.apply(x, self.lambda_val)


# ============================================================
# Material Classifier
# ============================================================

class MaterialClassifier(nn.Module):
    """材質分類器，接在 Encoder features 之後。

    透過 Global Average Pooling + MLP 將 encoder features
    分類為 box/papercup/plastic。

    Args:
        feat_dim: Encoder feature 維度。
        hidden_dim: 隱藏層維度。
        num_classes: 材質類別數。
        dropout: Dropout 機率。
    """

    def __init__(
        self,
        feat_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = NUM_MATERIALS,
        dropout: float = 0.3,
    ):
        """初始化 MaterialClassifier。

        Args:
            feat_dim: Encoder feature 維度。
            hidden_dim: 隱藏層維度。
            num_classes: 材質類別數。
            dropout: Dropout 機率。
        """
        super().__init__()
        self.grl = GradientReversalLayer(lambda_val=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def set_lambda(self, lambda_val: float):
        """動態設定梯度反轉係數。

        Args:
            lambda_val: 新的梯度反轉係數。
        """
        self.grl.lambda_val = lambda_val

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """前向傳播。

        Args:
            encoder_features: Encoder 輸出 (B, C, T)。

        Returns:
            材質分類 logits (B, num_classes)。
        """
        # Global Average Pooling over time dimension
        pooled = encoder_features.mean(dim=-1)  # (B, C)
        reversed_features = self.grl(pooled)
        return self.classifier(reversed_features)


# ============================================================
# Losses (same as train_material_gen.py)
# ============================================================

class STFTLoss(nn.Module):
    """單一解析度 STFT Loss。

    Args:
        n_fft: FFT 大小。
        hop_length: Hop 大小。
        win_length: 窗口大小。
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        """初始化 STFTLoss。

        Args:
            n_fft: FFT 大小。
            hop_length: Hop 大小。
            win_length: 窗口大小。
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def _stft(self, x):
        """計算 STFT 取幅度。

        Args:
            x: 輸入波形。

        Returns:
            STFT 幅度。
        """
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        )
        return torch.abs(spec)

    def forward(self, y_hat, y):
        """計算 SC + Log-Mag Loss。

        Args:
            y_hat: 預測波形。
            y: 目標波形。

        Returns:
            (SC Loss, Log-Mag Loss) 二元組。
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
        fft_sizes: FFT 大小列表。
        hop_sizes: Hop 大小列表。
        win_sizes: 窗口大小列表。
    """

    def __init__(self, fft_sizes=[2048, 1024, 512], hop_sizes=[512, 256, 128],
                 win_sizes=[2048, 1024, 512]):
        """初始化 MultiResolutionSTFTLoss。

        Args:
            fft_sizes: FFT 大小列表。
            hop_sizes: Hop 大小列表。
            win_sizes: 窗口大小列表。
        """
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(n, h, w) for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, y_hat, y):
        """計算多解析度 STFT Loss。

        Args:
            y_hat: 預測波形。
            y: 目標波形。

        Returns:
            (平均 SC Loss, 平均 Log-Mag Loss)。
        """
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        sc, mag = 0.0, 0.0
        for loss_fn in self.stft_losses:
            s, m = loss_fn(y_hat, y)
            sc += s
            mag += m
        n = len(self.stft_losses)
        return sc / n, mag / n


class MelReconstructionLoss(nn.Module):
    """Mel 頻譜重建 Loss。

    Args:
        sample_rate: 採樣率。
        n_fft: FFT 大小。
        hop_length: Hop 大小。
        n_mels: Mel 頻帶數量。
    """

    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        """初始化 MelReconstructionLoss。

        Args:
            sample_rate: 採樣率。
            n_fft: FFT 大小。
            hop_length: Hop 大小。
            n_mels: Mel 頻帶數。
        """
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, center=True, power=1,
        )

    def forward(self, y_hat, y):
        """計算 Log-Mel L1 Loss。

        Args:
            y_hat: 預測波形。
            y: 目標波形。

        Returns:
            Mel L1 Loss。
        """
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        mel_hat = torch.log(self.mel_spec(y_hat).clamp(min=1e-7))
        mel = torch.log(self.mel_spec(y).clamp(min=1e-7))
        return F.l1_loss(mel, mel_hat)


# ============================================================
# Utilities (same as train_material_gen.py)
# ============================================================

class _TeeIO:
    """同時輸出到多個 stream。"""
    def __init__(self, *streams):
        """初始化。

        Args:
            *streams: stream 列表。
        """
        self._streams = streams

    def write(self, data):
        """寫入。

        Args:
            data: 資料。
        """
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        """清空緩衝區。"""
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        """非終端。

        Returns:
            False。
        """
        return False


def setup_logging(output_dir: Path) -> Path:
    """設定日誌。

    Args:
        output_dir: 日誌目錄。

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
    """設定隨機種子。

    Args:
        seed: 種子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_preinit(device, retries=10, sleep_s=2.0):
    """CUDA 預初始化。

    Args:
        device: CUDA 裝置。
        retries: 重試次數。
        sleep_s: 重試間隔。
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


def compute_grl_lambda(epoch, max_epoch, gamma=10.0):
    """計算 GRL lambda，隨訓練進度漸進增加。

    使用 DANN paper 的 schedule: lambda = 2 / (1 + exp(-gamma * p)) - 1
    其中 p = epoch / max_epoch。

    Args:
        epoch: 當前 epoch。
        max_epoch: 總 epoch 數。
        gamma: 增長速率。

    Returns:
        當前 GRL lambda 值，範圍 [0, 1]。
    """
    p = epoch / max_epoch
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# ============================================================
# Material-Aware Dataset (extends MaterialAugDataset)
# ============================================================

class MaterialLabeledDataset(MaterialAugDataset):
    """帶有材質標籤的資料集，用於對抗訓練。

    在 MaterialAugDataset 基礎上返回 material_idx。

    Args:
        cache_path: 快取路徑。
        **kwargs: 傳遞給 MaterialAugDataset 的參數。
    """

    def __getitem__(self, idx):
        """取得資料，附加材質標籤。

        Args:
            idx: 樣本索引。

        Returns:
            包含 'material_idx' 的字典。
        """
        item = super().__getitem__(idx)

        # 從原始快取取得 material（父類使用 self.samples 儲存原始資料）
        raw = self.samples[idx]
        material = raw.get('material', 'unknown')
        item['material_idx'] = MATERIAL_TO_IDX.get(material, -1)
        item['material'] = material

        return item


def collate_fn_material(batch):
    """帶材質標籤的 collate 函式。

    Args:
        batch: 樣本列表。

    Returns:
        包含 'material_idx' tensor 的批次字典。
    """
    base_batch = collate_fn_curriculum(batch)

    material_idxs = [item.get('material_idx', -1) for item in batch]
    base_batch['material_idx'] = torch.tensor(material_idxs, dtype=torch.long)

    return base_batch


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, material_clf, dataloader, optimizer, clf_optimizer,
                device, epoch, config, mr_stft_loss_fn, mel_loss_fn,
                scaler=None) -> Dict:
    """執行一個 epoch 的對抗訓練。

    Args:
        model: TeacherStudentNoVQ 模型。
        material_clf: MaterialClassifier。
        dataloader: 訓練用 DataLoader。
        optimizer: 主模型優化器。
        clf_optimizer: 分類器優化器。
        device: 計算裝置。
        epoch: 當前 epoch。
        config: 訓練設定。
        mr_stft_loss_fn: MR-STFT Loss。
        mel_loss_fn: Mel Loss。
        scaler: AMP GradScaler。

    Returns:
        各項 loss 平均值字典。
    """
    model.train()
    model.teacher.backbone.eval()
    model.teacher.head.eval()
    model.student.train()
    material_clf.train()

    # 更新 GRL lambda
    grl_lambda = compute_grl_lambda(epoch, config['epochs'])
    material_clf.set_lambda(grl_lambda)

    metrics = {
        'total_loss': 0.0, 'wav_mse': 0.0,
        'stft_sc_loss': 0.0, 'stft_mag_loss': 0.0,
        'mel_loss': 0.0, 'feat_align_loss': 0.0,
        'adv_loss': 0.0, 'material_acc': 0.0,
        'nan_batches': 0, 'grl_lambda': grl_lambda,
    }
    n_batches = 0
    nan_count = 0
    correct_materials = 0
    total_materials = 0

    lambda_wav = config['lambda_wav']
    lambda_stft = config['lambda_stft']
    lambda_mel = config['lambda_mel']
    lambda_feat = config['lambda_feat']
    lambda_adv = config['lambda_adv']

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [mat-adv λ_grl={grl_lambda:.3f}]")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        material_idx = batch['material_idx'].to(device)  # (B,)

        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        # 過濾掉 unknown 材質 (idx == -1)
        valid_mask = material_idx >= 0
        has_material_labels = valid_mask.sum().item() > 0

        if batch_idx % config['grad_accum'] == 0:
            optimizer.zero_grad()
            clf_optimizer.zero_grad()

        with autocast(enabled=config['use_amp']):
            out = model.forward_wav(clean_audio, noisy_audio)
            recon_wav = out['recon_wav']
            student_feat = out['student_encoder_out']
            teacher_feat = out['teacher_encoder_out']

            T = min(clean_audio.shape[-1], recon_wav.shape[-1])
            recon_t = recon_wav[..., :T]
            clean_t = clean_audio[..., :T]

            # 重建 Loss（同方法 1+2）
            wav_mse = F.mse_loss(recon_t, clean_t)
            sc_loss, mag_loss = mr_stft_loss_fn(recon_t, clean_t)
            stft_loss = sc_loss + mag_loss
            mel_loss = mel_loss_fn(recon_t, clean_t)

            Tf = min(student_feat.shape[-1], teacher_feat.shape[-1])
            feat_align = F.mse_loss(
                student_feat[..., :Tf],
                teacher_feat[..., :Tf].detach(),
            )

            recon_loss = (
                lambda_wav * wav_mse
                + lambda_stft * stft_loss
                + lambda_mel * mel_loss
                + lambda_feat * feat_align
            )

            # 對抗 Loss（只在有材質標籤時計算）
            adv_loss = torch.tensor(0.0, device=device)
            if has_material_labels:
                material_logits = material_clf(student_feat[valid_mask])
                adv_loss = F.cross_entropy(
                    material_logits, material_idx[valid_mask]
                )

                # 統計分類準確率
                pred = material_logits.argmax(dim=-1)
                correct_materials += (pred == material_idx[valid_mask]).sum().item()
                total_materials += valid_mask.sum().item()

            loss = (recon_loss + lambda_adv * adv_loss) / config['grad_accum']

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            metrics['nan_batches'] = nan_count
            optimizer.zero_grad()
            clf_optimizer.zero_grad()
            if nan_count >= 10:
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
                # clf_optimizer 只在有材質標籤產生梯度時才 step
                if has_material_labels:
                    scaler.unscale_(clf_optimizer)
                    scaler.step(clf_optimizer)
                scaler.update()
        else:
            loss.backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip'],
                )
                optimizer.step()
                if has_material_labels:
                    clf_optimizer.step()

        loss_val = loss.item() * config['grad_accum']
        metrics['total_loss'] += loss_val
        metrics['wav_mse'] += wav_mse.item()
        metrics['stft_sc_loss'] += sc_loss.item()
        metrics['stft_mag_loss'] += mag_loss.item()
        metrics['mel_loss'] += mel_loss.item()
        metrics['feat_align_loss'] += feat_align.item()
        metrics['adv_loss'] += adv_loss.item()
        n_batches += 1

        pbar.set_postfix({
            'total': f"{loss_val:.4f}",
            'adv': f"{adv_loss.item():.4f}",
            'mat_acc': f"{100*correct_materials/max(total_materials,1):.1f}%",
        })

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
                   'mel_loss', 'feat_align_loss', 'adv_loss']:
            metrics[k] /= n_batches

    metrics['material_acc'] = correct_materials / max(total_materials, 1)
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, config,
             mr_stft_loss_fn, mel_loss_fn, max_batches=30) -> Dict:
    """在驗證集上評估模型。

    Args:
        model: 模型。
        dataloader: 驗證用 DataLoader。
        device: 計算裝置。
        config: 設定字典。
        mr_stft_loss_fn: MR-STFT Loss。
        mel_loss_fn: Mel Loss。
        max_batches: 最多評估幾個 batch。

    Returns:
        驗證 loss 字典。
    """
    model.eval()
    wav_mse_list, noisy_mse_list = [], []
    stft_sc_list, stft_mag_list, mel_list = [], [], []
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

        Tf = min(student_feat.shape[-1], teacher_feat.shape[-1])
        feat_align_list.append(
            F.mse_loss(student_feat[..., :Tf], teacher_feat[..., :Tf]).item()
        )

    model.train()
    return {
        'val_wav_mse':    float(np.mean(wav_mse_list))    if wav_mse_list else float('nan'),
        'val_noisy_mse':  float(np.mean(noisy_mse_list))  if noisy_mse_list else float('nan'),
        'val_stft_sc':    float(np.mean(stft_sc_list))    if stft_sc_list else float('nan'),
        'val_stft_mag':   float(np.mean(stft_mag_list))   if stft_mag_list else float('nan'),
        'val_mel_loss':   float(np.mean(mel_list))        if mel_list else float('nan'),
        'val_feat_align': float(np.mean(feat_align_list)) if feat_align_list else float('nan'),
    }


def _save_audio_samples(model, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    """儲存音訊樣本。

    Args:
        model: 模型。
        loader: DataLoader。
        device: 裝置。
        output_dir: 輸出目錄。
        epoch: 當前 epoch。
        num_samples: 樣本數。
        split: 'train' 或 'val'。
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
                    """內部儲存函式。

                    Args:
                        tensor: 音訊張量。
                        name: 檔案名。
                    """
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
    """繪製含對抗 loss 的訓練曲線。

    Args:
        history: 訓練歷程。
        output_dir: 輸出目錄。
        epoch: 當前 epoch。
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f'exp_0304-adv: Material Adversarial (Epoch {epoch})', fontsize=14)
    epochs = range(1, len(history['train_total_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_total_loss'], 'b-', alpha=0.8)
    axes[0, 0].set_title('Total Loss (train)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)

    if history.get('val_wav_mse'):
        axes[0, 1].plot(epochs, history['val_wav_mse'], 'r-', label='Recon', alpha=0.8)
    if history.get('val_noisy_mse'):
        axes[0, 1].plot(epochs, history['val_noisy_mse'], 'gray', ls='--', label='Noisy', alpha=0.8)
    axes[0, 1].set_title('Wav MSE (val)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    if history.get('train_feat_align'):
        axes[1, 0].plot(epochs, history['train_feat_align'], 'b-', label='Train', alpha=0.8)
    if history.get('val_feat_align'):
        axes[1, 0].plot(epochs, history['val_feat_align'], 'r--', label='Val', alpha=0.8)
    axes[1, 0].set_title('Feature Alignment Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 對抗 Loss
    if history.get('train_adv_loss'):
        axes[1, 1].plot(epochs, history['train_adv_loss'], 'orange', alpha=0.8)
    axes[1, 1].set_title('Adversarial Loss (material classification)')
    axes[1, 1].grid(True)

    # 分類準確率（期望下降到 1/3 ≈ 33%）
    if history.get('train_material_acc'):
        axes[2, 0].plot(epochs, [a * 100 for a in history['train_material_acc']],
                        'purple', alpha=0.8)
        axes[2, 0].axhline(y=100 / NUM_MATERIALS, color='red', ls='--',
                           label=f'Chance ({100/NUM_MATERIALS:.0f}%)', alpha=0.7)
    axes[2, 0].set_title('Material Classification Acc (%)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # GRL lambda
    if history.get('grl_lambda'):
        axes[2, 1].plot(epochs, history['grl_lambda'], 'green', linewidth=2)
    axes[2, 1].set_title('GRL Lambda (gradient reversal strength)')
    axes[2, 1].grid(True)

    if history.get('lr'):
        axes[3, 0].plot(epochs, history['lr'], 'green', linewidth=2)
    axes[3, 0].set_title('Learning Rate')
    axes[3, 0].grid(True)

    axes[3, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch{epoch:03d}.png', dpi=150)
    plt.close()
    print(f"  Loss plot saved: training_curves_epoch{epoch:03d}.png")


# ============================================================
# Main
# ============================================================

def main():
    """exp_0304 備用方案 (方法 3): 材質對抗訓練。

    在方法 1+2 的材質增強基礎上，加入 Gradient Reversal + Material Classifier，
    讓 Encoder LoRA 學到材質不變的特徵表示。
    """
    parser = argparse.ArgumentParser(
        description='exp_0304: Material Adversarial Training (Method 3 Backup)'
    )

    parser.add_argument('--mode', type=str, default='smoke',
                        choices=['smoke', 'epoch'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--encoder_ckpt', type=str, default=EXP0227_BEST_CKPT)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--clf_learning_rate', type=float, default=1e-4,
                        help='Material classifier 學習率')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)

    # Loss weights
    parser.add_argument('--lambda_wav', type=float, default=1.0)
    parser.add_argument('--lambda_stft', type=float, default=1.0)
    parser.add_argument('--lambda_mel', type=float, default=45.0)
    parser.add_argument('--lambda_feat', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=0.1,
                        help='對抗 loss 權重（建議從小開始）')

    parser.add_argument('--stft_fft_sizes', type=str, default='2048,1024,512')
    parser.add_argument('--stft_hop_sizes', type=str, default='512,256,128')
    parser.add_argument('--stft_win_sizes', type=str, default='2048,1024,512')

    # 材質分類器參數
    parser.add_argument('--clf_hidden_dim', type=int, default=256)
    parser.add_argument('--clf_dropout', type=float, default=0.3)
    parser.add_argument('--grl_gamma', type=float, default=10.0,
                        help='GRL lambda schedule gamma')

    # 原有增強
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

    # 材質增強
    parser.add_argument('--freq_response_prob', type=float, default=0.5)
    parser.add_argument('--freq_response_n_bands_min', type=int, default=2)
    parser.add_argument('--freq_response_n_bands_max', type=int, default=5)
    parser.add_argument('--freq_response_gain_db', type=float, default=10.0)
    parser.add_argument('--spectral_norm_prob', type=float, default=0.3)
    parser.add_argument('--random_lowpass_prob', type=float, default=0.3)
    parser.add_argument('--random_lowpass_min', type=float, default=2000.0)
    parser.add_argument('--random_lowpass_max', type=float, default=6000.0)
    parser.add_argument('--resonance_prob', type=float, default=0.3)
    parser.add_argument('--resonance_n_peaks_min', type=int, default=1)
    parser.add_argument('--resonance_n_peaks_max', type=int, default=3)

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
        exp_dir = Path(f'exp_0304/runs/material_adv_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0304_material_adversarial'
    config['encoder_init'] = f'exp_0227 best: {args.encoder_ckpt}'
    config['method'] = ('Method 1+2+3: FreqResponse + SpectralNorm + '
                        'Domain Adversarial Training')

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0304-adv: Material Adversarial Training (Backup Method 3)")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"λ_adv={args.lambda_adv}, GRL gamma={args.grl_gamma}")
    print(f"Material classes: {MATERIAL_TO_IDX}")
    print(f"Encoder init: {args.encoder_ckpt}")
    print(f"Output: {exp_dir}")
    print("=" * 70)

    device = torch.device(args.device)
    cuda_preinit(device)

    # ==== Data ====
    print("\nLoading data (material-labeled + augmented)...")
    if args.mode == 'smoke':
        full_ds = MaterialLabeledDataset(
            VAL_CACHE, augment=True,
            freq_response_prob=args.freq_response_prob,
            freq_response_n_bands=(args.freq_response_n_bands_min, args.freq_response_n_bands_max),
            spectral_norm_prob=args.spectral_norm_prob,
            random_lowpass_prob=args.random_lowpass_prob,
            resonance_prob=args.resonance_prob,
            sample_rate=SAMPLE_RATE,
            filter_clean_to_clean=True, compute_snr=False,
        )
        smoke_indices = list(range(min(20, len(full_ds))))
        smoke_ds = Subset(full_ds, smoke_indices)
        train_loader = DataLoader(
            smoke_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn_material,
        )

        val_ds = MaterialLabeledDataset(
            VAL_CACHE, augment=False,
            filter_clean_to_clean=True, compute_snr=False,
        )
        val_smoke = Subset(val_ds, smoke_indices)
        val_loader = DataLoader(
            val_smoke, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn_material,
        )
        print(f"Smoke test: {len(smoke_ds)} samples")
    else:
        train_ds = MaterialLabeledDataset(
            TRAIN_CACHE, augment=True,
            snr_remix_prob=args.snr_remix_prob,
            snr_remix_range=(args.snr_remix_min, args.snr_remix_max),
            random_gain_prob=args.random_gain_prob,
            random_gain_db=args.random_gain_db,
            random_crop_prob=args.random_crop_prob,
            random_crop_min_ratio=args.random_crop_min_ratio,
            time_stretch_prob=args.time_stretch_prob,
            time_stretch_range=(args.time_stretch_min, args.time_stretch_max),
            freq_response_prob=args.freq_response_prob,
            freq_response_n_bands=(args.freq_response_n_bands_min, args.freq_response_n_bands_max),
            freq_response_gain_db=args.freq_response_gain_db,
            spectral_norm_prob=args.spectral_norm_prob,
            random_lowpass_prob=args.random_lowpass_prob,
            random_lowpass_range=(args.random_lowpass_min, args.random_lowpass_max),
            resonance_prob=args.resonance_prob,
            resonance_n_peaks=(args.resonance_n_peaks_min, args.resonance_n_peaks_max),
            sample_rate=SAMPLE_RATE,
            filter_clean_to_clean=True, compute_snr=False,
        )
        val_ds = MaterialLabeledDataset(
            VAL_CACHE, augment=False,
            filter_clean_to_clean=True, compute_snr=False,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=2, collate_fn=collate_fn_material,
        )
        val_loader = DataLoader(
            val_ds, batch_size=4, shuffle=False,
            num_workers=2, collate_fn=collate_fn_material,
        )
        print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ==== Model ====
    print("\nBuilding TeacherStudentNoVQ + MaterialClassifier...")
    model = TeacherStudentNoVQ(
        wavtok_config=WAVTOK_CONFIG, wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6], device=device,
    ).to(device)

    # Load encoder LoRA weights
    ckpt_path = Path(args.encoder_ckpt)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"  Loaded model_state_dict from {ckpt_path}")
        elif 'encoder_lora_state' in ckpt:
            model.student.load_state_dict(ckpt['encoder_lora_state'], strict=False)
            print(f"  Loaded encoder_lora_state from {ckpt_path}")
        print(f"  Source epoch: {ckpt.get('epoch', '?')}")
    else:
        print(f"  WARNING: encoder ckpt not found: {ckpt_path}")

    # Material Classifier
    material_clf = MaterialClassifier(
        feat_dim=512, hidden_dim=args.clf_hidden_dim,
        num_classes=NUM_MATERIALS, dropout=args.clf_dropout,
    ).to(device)
    print(f"  MaterialClassifier: {sum(p.numel() for p in material_clf.parameters()):,} params")

    # ==== Losses ====
    mr_stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_sizes=win_sizes,
    ).to(device)
    mel_loss_fn = MelReconstructionLoss(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100,
    ).to(device)

    # ==== Optimizers ====
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    clf_optimizer = torch.optim.AdamW(
        material_clf.parameters(), lr=args.clf_learning_rate,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        """Warmup + Cosine decay LR schedule。

        Args:
            epoch: 當前 epoch。

        Returns:
            學習率倍率。
        """
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.learning_rate,
                   0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if args.use_amp else None

    best_val_total = float('inf')
    history = {
        'train_total_loss': [], 'train_wav_mse': [],
        'train_stft_sc': [], 'train_stft_mag': [], 'train_mel': [],
        'train_feat_align': [], 'train_adv_loss': [],
        'train_material_acc': [], 'grl_lambda': [],
        'val_wav_mse': [], 'val_noisy_mse': [],
        'val_stft_sc': [], 'val_stft_mag': [], 'val_mel_loss': [],
        'val_feat_align': [], 'lr': [],
    }

    print("\nStarting adversarial training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, material_clf, train_loader, optimizer, clf_optimizer,
            device, epoch, config, mr_stft_loss_fn, mel_loss_fn, scaler,
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
        history['train_adv_loss'].append(train_metrics['adv_loss'])
        history['train_material_acc'].append(train_metrics['material_acc'])
        history['grl_lambda'].append(train_metrics['grl_lambda'])
        history['val_wav_mse'].append(val_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(val_metrics.get('val_noisy_mse', float('nan')))
        history['val_stft_sc'].append(val_metrics['val_stft_sc'])
        history['val_stft_mag'].append(val_metrics['val_stft_mag'])
        history['val_mel_loss'].append(val_metrics['val_mel_loss'])
        history['val_feat_align'].append(val_metrics['val_feat_align'])
        history['lr'].append(current_lr)

        val_total = (val_metrics['val_wav_mse']
                     + val_metrics['val_stft_sc']
                     + val_metrics['val_stft_mag']
                     + args.lambda_mel * val_metrics['val_mel_loss'])

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: total={train_metrics['total_loss']:.4f}  "
              f"adv={train_metrics['adv_loss']:.4f}  "
              f"mat_acc={100*train_metrics['material_acc']:.1f}%  "
              f"grl_λ={train_metrics['grl_lambda']:.3f}")
        print(f"  Val:   val_total={val_total:.4f}  "
              f"mse={val_metrics['val_wav_mse']:.5f}  "
              f"feat={val_metrics['val_feat_align']:.5f}")
        print(f"  LR={current_lr:.3e}")

        if val_total < best_val_total:
            best_val_total = val_total
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'material_clf_state': material_clf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'clf_optimizer_state': clf_optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, exp_dir / 'best_model_val_total.pt')
            print(f"  ★ New best val_total: {best_val_total:.4f}")

        if epoch % args.save_checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'material_clf_state': material_clf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'clf_optimizer_state': clf_optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, exp_dir / f'checkpoint_epoch{epoch:03d}.pt')

        if epoch % args.save_audio_interval == 0 or epoch == args.epochs:
            plot_training_curves(history, exp_dir, epoch)
            _save_audio_samples(model, val_loader, device, exp_dir, epoch)

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nAdversarial training complete. Best val_total: {best_val_total:.4f}")
    print(f"Output: {exp_dir}")


if __name__ == '__main__':
    main()

"""
exp_0304: 材質泛化 Encoder LoRA 訓練 — 方法 1+2（隨機頻率響應 + 頻譜正規化）

架構（同 exp_0226a）：
    Encoder LoRA (Trainable, 初始化自 exp_0227 best_model_val_total.pt):
        Noisy Audio → LoRA Encoder → student_encoder_out [B, 512, T]
    Decoder (Frozen, WavTokenizer pretrained):
        student_encoder_out → backbone → head → recon_wav

新增增強（在 exp_0216 的 4 種增強基礎上）：
    5. Random Frequency Response — 模擬未知材質 h(f)
    6. Spectral Envelope Normalization — 正規化到 canonical LDV 分佈
    7. Random Low-pass — 模擬不同材質高頻衰減
    8. Resonance Injection — 模擬材質機械共振

Loss（擴展自 exp_0226a）：
    λ_wav  * MSE(recon_wav, clean_wav)
    + λ_stft * MR-STFT(recon_wav, clean_wav)
    + λ_mel  * Mel(recon_wav, clean_wav)
    + λ_feat * MSE(student_feat, teacher_feat)           # 最終層 feature alignment
    + λ_inter * mean(MSE(student_L_i, teacher_L_i))      # 中間層 feature alignment (L3,L4,L6)
    + λ_pre_istft * MSE(student_stft, teacher_stft)      # pre-ISTFT STFT 域 loss（可選）

設計動機：
    先前實驗（exp_0224a）僅在 box/papercup/plastic 三種材質訓練，
    LoRA 學到「LDV 頻譜分佈 → 語音頻譜分佈」的映射。
    但在 mac（未知材質）上效果有限，且完全無法處理 AWGN。
    透過模擬不同材質的頻率響應和共振特性，
    讓模型學到更通用的「任意 LDV 頻譜 → 語音」映射。

執行：
    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python families/official/material_generalization/train_material_gen.py --mode smoke

    PYTHONPATH=/home/sbplab/ruizi/WavTokenize-self-supervised:$PYTHONPATH \\
    /home/sbplab/miniconda3/envs/test/bin/python families/official/material_generalization/train_material_gen.py \\
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
from typing import Dict, List, Tuple
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from families.deps.no_vq_core.models_no_vq import TeacherStudentNoVQ
from families.official.material_generalization.data_material_aug import MaterialAugDataset, create_material_aug_dataloaders
from families.deps.encoder_aug.data_augmented import collate_fn_curriculum
from utils.audio_losses import MelReconstructionLoss, MultiResolutionSTFTLoss
from utils.train_runtime import cuda_preinit, set_seed, setup_logging


# ============================================================
# Constants
# ============================================================

SAMPLE_RATE = 24000

EXP0227_BEST_CKPT = (
    'families/deps/enc_mrd_fm/runs/enc_mrd_fm_epoch_20260227_024953/best_model_val_total.pt'
)


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, dataloader, optimizer, device, epoch, config,
                mr_stft_loss_fn, mel_loss_fn, scaler=None) -> Dict:
    """執行一個 epoch 的訓練。

    Args:
        model: TeacherStudentNoVQ 模型。
        dataloader: 訓練用 DataLoader。
        optimizer: 優化器。
        device: 計算裝置。
        epoch: 當前 epoch 數。
        config: 訓練設定字典。
        mr_stft_loss_fn: MR-STFT Loss 函式。
        mel_loss_fn: Mel Loss 函式。
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
        'inter_feat_loss': 0.0,
        'pre_istft_loss': 0.0,
        'nan_batches': 0,
    }
    n_batches = 0
    nan_count = 0
    max_nan_per_epoch = 10

    lambda_wav = config['lambda_wav']
    lambda_stft = config['lambda_stft']
    lambda_mel = config['lambda_mel']
    lambda_feat = config['lambda_feat']
    lambda_inter_feat = config.get('lambda_inter_feat', 0.0)
    lambda_pre_istft = config.get('lambda_pre_istft', 0.0)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [material-gen]")

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
            out = model.forward_wav(
                clean_audio, noisy_audio,
                return_intermediates=(lambda_inter_feat > 0),
                return_pre_istft=(lambda_pre_istft > 0),
            )
            recon_wav = out['recon_wav']
            student_feat = out['student_encoder_out']
            teacher_feat = out['teacher_encoder_out']

            T = min(clean_audio.shape[-1], recon_wav.shape[-1])
            recon_t = recon_wav[..., :T]
            clean_t = clean_audio[..., :T]

            wav_mse = F.mse_loss(recon_t, clean_t)
            sc_loss, mag_loss = mr_stft_loss_fn(recon_t, clean_t)
            stft_loss = sc_loss + mag_loss
            mel_loss = mel_loss_fn(recon_t, clean_t)

            Tf = min(student_feat.shape[-1], teacher_feat.shape[-1])
            feat_align = F.mse_loss(
                student_feat[..., :Tf],
                teacher_feat[..., :Tf].detach(),
            )

            # Intermediate feature alignment loss (layers 3, 4, 6)
            inter_feat = torch.tensor(0.0, device=device)
            if lambda_inter_feat > 0 and 'student_intermediates' in out:
                s_inters = out['student_intermediates']
                t_inters = out['teacher_intermediates']
                n_layers = 0
                for idx in s_inters:
                    if idx in t_inters:
                        s_i = s_inters[idx]
                        t_i = t_inters[idx]
                        Ti = min(s_i.shape[-1], t_i.shape[-1])
                        inter_feat = inter_feat + F.mse_loss(
                            s_i[..., :Ti], t_i[..., :Ti].detach())
                        n_layers += 1
                if n_layers > 0:
                    inter_feat = inter_feat / n_layers

            # Pre-ISTFT STFT domain loss
            pre_istft = torch.tensor(0.0, device=device)
            if lambda_pre_istft > 0 and 'student_pre_istft' in out:
                s_stft = out['student_pre_istft']
                t_stft = out['teacher_pre_istft']
                Ts = min(s_stft.shape[-1], t_stft.shape[-1])
                pre_istft = F.mse_loss(
                    s_stft[..., :Ts], t_stft[..., :Ts].detach())

            loss = (
                lambda_wav * wav_mse
                + lambda_stft * stft_loss
                + lambda_mel * mel_loss
                + lambda_feat * feat_align
                + lambda_inter_feat * inter_feat
                + lambda_pre_istft * pre_istft
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
        metrics['inter_feat_loss'] += inter_feat.item()
        metrics['pre_istft_loss'] += pre_istft.item()
        n_batches += 1

        pbar.set_postfix({
            'total': f"{loss_val:.4f}",
            'wav': f"{wav_mse.item():.5f}",
            'feat': f"{feat_align.item():.5f}",
            'mel': f"{mel_loss.item():.3f}",
        })

    if n_batches > 0:
        for k in ['total_loss', 'wav_mse', 'stft_sc_loss', 'stft_mag_loss',
                   'mel_loss', 'feat_align_loss', 'inter_feat_loss',
                   'pre_istft_loss']:
            metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, config,
             mr_stft_loss_fn, mel_loss_fn, max_batches=30) -> Dict:
    """在驗證集上評估模型。

    Args:
        model: TeacherStudentNoVQ 模型。
        dataloader: 驗證用 DataLoader。
        device: 計算裝置。
        config: 訓練設定字典。
        mr_stft_loss_fn: MR-STFT Loss 函式。
        mel_loss_fn: Mel Loss 函式。
        max_batches: 最多評估幾個 batch。

    Returns:
        包含各項驗證 loss 的字典。
    """
    model.eval()

    lambda_inter_feat = config.get('lambda_inter_feat', 0.0)
    lambda_pre_istft = config.get('lambda_pre_istft', 0.0)

    wav_mse_list, noisy_mse_list = [], []
    stft_sc_list, stft_mag_list, mel_list = [], [], []
    noisy_stft_sc_list, noisy_mel_list = [], []
    feat_align_list = []
    inter_feat_list = []
    pre_istft_list = []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        out = model.forward_wav(
            clean_audio, noisy_audio,
            return_intermediates=(lambda_inter_feat > 0),
            return_pre_istft=(lambda_pre_istft > 0),
        )
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

        # Intermediate feature alignment
        if lambda_inter_feat > 0 and 'student_intermediates' in out:
            s_inters = out['student_intermediates']
            t_inters = out['teacher_intermediates']
            layer_losses = []
            for idx in s_inters:
                if idx in t_inters:
                    s_i = s_inters[idx]
                    t_i = t_inters[idx]
                    Ti = min(s_i.shape[-1], t_i.shape[-1])
                    layer_losses.append(
                        F.mse_loss(s_i[..., :Ti], t_i[..., :Ti]).item())
            if layer_losses:
                inter_feat_list.append(float(np.mean(layer_losses)))

        # Pre-ISTFT loss
        if lambda_pre_istft > 0 and 'student_pre_istft' in out:
            s_stft = out['student_pre_istft']
            t_stft = out['teacher_pre_istft']
            Ts = min(s_stft.shape[-1], t_stft.shape[-1])
            pre_istft_list.append(
                F.mse_loss(s_stft[..., :Ts], t_stft[..., :Ts]).item())

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
        'val_inter_feat':    float(np.mean(inter_feat_list))    if inter_feat_list else float('nan'),
        'val_pre_istft':     float(np.mean(pre_istft_list))     if pre_istft_list else float('nan'),
    }


def _save_audio_samples(model, loader, device, output_dir, epoch,
                        num_samples=2, split='val'):
    """儲存音訊樣本供人工檢聽。

    Args:
        model: 模型。
        loader: DataLoader。
        device: 計算裝置。
        output_dir: 輸出目錄。
        epoch: 當前 epoch。
        num_samples: 儲存樣本數。
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
                    """儲存音訊張量為 wav 檔。

                    Args:
                        tensor: 音訊張量。
                        name: 檔案名稱。
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
    """繪製訓練曲線圖。

    Args:
        history: 訓練歷程字典。
        output_dir: 圖片輸出目錄。
        epoch: 當前 epoch。
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'exp_0304: Material Generalization (Epoch {epoch})', fontsize=14)

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
    ax.set_title('STFT Losses')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1, 1]
    if history.get('train_feat_align'):
        ax.plot(epochs, history['train_feat_align'], 'b-', label='Train feat align', alpha=0.8)
    if history.get('val_feat_align'):
        ax.plot(epochs, history['val_feat_align'], 'r--', label='Val feat align', alpha=0.8)
    ax.set_title('Feature Alignment Loss')
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
        ax.set_title('Val Total Loss')
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
    """exp_0304 主訓練流程：材質泛化 Encoder LoRA。

    使用方法 1+2（隨機頻率響應增強 + 頻譜正規化）的 MaterialAugDataset，
    訓練 Encoder LoRA 在 feature alignment 框架下學習
    更通用的 LDV → 語音映射。

    初始化自 exp_0227 best 權重。
    """
    parser = argparse.ArgumentParser(
        description='exp_0304: Material Generalization — Encoder LoRA + FreqAug + SpectralNorm'
    )

    parser.add_argument('--mode', type=str, default='smoke',
                        choices=['smoke', 'epoch'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--encoder_ckpt', type=str, default=EXP0227_BEST_CKPT,
                        help='Encoder LoRA init checkpoint')
    parser.add_argument('--train_cache', type=str, default=str(TRAIN_CACHE),
                        help='Training cache path')
    parser.add_argument('--val_cache', type=str, default=str(VAL_CACHE),
                        help='Validation cache path')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
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
    parser.add_argument('--lambda_inter_feat', type=float, default=0.5,
                        help='Intermediate feature alignment loss weight (layers 3,4,6)')
    parser.add_argument('--lambda_pre_istft', type=float, default=0.0,
                        help='Pre-ISTFT STFT domain loss weight (0 to disable)')

    parser.add_argument('--stft_fft_sizes', type=str, default='2048,1024,512')
    parser.add_argument('--stft_hop_sizes', type=str, default='512,256,128')
    parser.add_argument('--stft_win_sizes', type=str, default='2048,1024,512')

    # 原有增強參數
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

    # 材質增強參數
    parser.add_argument('--freq_response_prob', type=float, default=0.5,
                        help='Random Frequency Response 觸發機率')
    parser.add_argument('--freq_response_n_bands_min', type=int, default=2)
    parser.add_argument('--freq_response_n_bands_max', type=int, default=5)
    parser.add_argument('--freq_response_gain_db', type=float, default=10.0,
                        help='每個 EQ band 的最大增益 (dB)')
    parser.add_argument('--spectral_norm_prob', type=float, default=0.3,
                        help='Spectral Normalization 觸發機率')
    parser.add_argument('--random_lowpass_prob', type=float, default=0.3,
                        help='Random Low-pass 觸發機率')
    parser.add_argument('--random_lowpass_min', type=float, default=2000.0,
                        help='LP 最低截止頻率 (Hz)')
    parser.add_argument('--random_lowpass_max', type=float, default=6000.0,
                        help='LP 最高截止頻率 (Hz)')
    parser.add_argument('--resonance_prob', type=float, default=0.3,
                        help='Resonance Injection 觸發機率')
    parser.add_argument('--resonance_n_peaks_min', type=int, default=1)
    parser.add_argument('--resonance_n_peaks_max', type=int, default=3)

    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers (0 for single-process loading)')
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
        exp_dir = Path(f'families/official/material_generalization/runs/material_gen_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['experiment'] = 'exp_0304_material_generalization'
    config['encoder_init'] = f'exp_0227 best: {args.encoder_ckpt}'
    config['train_cache'] = args.train_cache
    config['val_cache'] = args.val_cache
    config['method'] = 'Random FreqResponse + Spectral Normalization (Method 1+2)'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0304: Material Generalization — FreqAug + SpectralNorm")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LR: {args.learning_rate} → {args.min_lr}")
    print(f"Encoder LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"VQ: SKIPPED  |  Decoder: FROZEN")
    print(f"Encoder init: {args.encoder_ckpt}")
    print(f"Loss: λ_wav={args.lambda_wav} + λ_stft={args.lambda_stft} "
          f"+ λ_mel={args.lambda_mel} + λ_feat={args.lambda_feat} "
          f"+ λ_inter={args.lambda_inter_feat} + λ_pre_istft={args.lambda_pre_istft}")
    print(f"Material Aug: freq_resp_p={args.freq_response_prob}, "
          f"spec_norm_p={args.spectral_norm_prob}, "
          f"lp_p={args.random_lowpass_prob}, "
          f"reso_p={args.resonance_prob}")
    print(f"Train cache: {args.train_cache}")
    print(f"Val cache: {args.val_cache}")
    print(f"Output: {exp_dir}")
    if log_path:
        print(f"Log: {log_path}")
    print("=" * 70)

    device = torch.device(args.device)
    cuda_preinit(device)

    # ==== Data ====
    print("\nLoading data (with material augmentation)...")
    train_cache = Path(args.train_cache)
    val_cache = Path(args.val_cache)
    if args.mode == 'smoke':
        full_ds = MaterialAugDataset(
            val_cache, augment=True,
            freq_response_prob=args.freq_response_prob,
            freq_response_n_bands=(args.freq_response_n_bands_min, args.freq_response_n_bands_max),
            freq_response_gain_db=args.freq_response_gain_db,
            spectral_norm_prob=args.spectral_norm_prob,
            random_lowpass_prob=args.random_lowpass_prob,
            random_lowpass_range=(args.random_lowpass_min, args.random_lowpass_max),
            resonance_prob=args.resonance_prob,
            resonance_n_peaks=(args.resonance_n_peaks_min, args.resonance_n_peaks_max),
            sample_rate=SAMPLE_RATE,
            filter_clean_to_clean=True,
            compute_snr=False,
        )
        smoke_indices = list(range(min(20, len(full_ds))))
        smoke_ds = Subset(full_ds, smoke_indices)
        train_loader = DataLoader(
            smoke_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn_curriculum,
        )
        val_ds = MaterialAugDataset(
            val_cache, augment=False,
            filter_clean_to_clean=True, compute_snr=False,
        )
        val_smoke = Subset(val_ds, smoke_indices)
        val_loader = DataLoader(
            val_smoke, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn_curriculum,
        )
        print(f"Smoke test: {len(smoke_ds)} samples")
    else:
        train_loader, val_loader = create_material_aug_dataloaders(
            train_cache, val_cache,
            batch_size=args.batch_size,
            sample_rate=SAMPLE_RATE,
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
        )
        print(f"Train: {len(train_loader.dataset)} samples, "
              f"Val: {len(val_loader.dataset)} samples")

    # ==== Model ====
    print("\nBuilding TeacherStudentNoVQ (encoder-only trainable)...")
    model = TeacherStudentNoVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6],
        device=device,
    ).to(device)

    # 載入 exp_0227 encoder LoRA 權重
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

    # ==== Losses ====
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

    # ==== Optimizer ====
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        """計算學習率倍率（warmup + cosine decay）。

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
    best_val_mse = float('inf')
    history = {
        'train_total_loss': [], 'train_wav_mse': [],
        'train_stft_sc': [], 'train_stft_mag': [], 'train_mel': [],
        'train_feat_align': [], 'train_inter_feat': [], 'train_pre_istft': [],
        'val_wav_mse': [], 'val_noisy_mse': [],
        'val_stft_sc': [], 'val_stft_mag': [], 'val_mel_loss': [],
        'val_noisy_stft_sc': [], 'val_noisy_mel': [],
        'val_feat_align': [], 'val_inter_feat': [], 'val_pre_istft': [],
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
        history['train_inter_feat'].append(train_metrics['inter_feat_loss'])
        history['train_pre_istft'].append(train_metrics['pre_istft_loss'])
        history['val_wav_mse'].append(val_metrics['val_wav_mse'])
        history['val_noisy_mse'].append(val_metrics['val_noisy_mse'])
        history['val_stft_sc'].append(val_metrics['val_stft_sc'])
        history['val_stft_mag'].append(val_metrics['val_stft_mag'])
        history['val_mel_loss'].append(val_metrics['val_mel_loss'])
        history['val_noisy_stft_sc'].append(val_metrics['val_noisy_stft_sc'])
        history['val_noisy_mel'].append(val_metrics['val_noisy_mel'])
        history['val_feat_align'].append(val_metrics['val_feat_align'])
        history['val_inter_feat'].append(val_metrics['val_inter_feat'])
        history['val_pre_istft'].append(val_metrics['val_pre_istft'])
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
              f"inter={train_metrics['inter_feat_loss']:.5f}  "
              f"mel={train_metrics['mel_loss']:.3f}")
        print(f"  Val:   recon_mse={val_mse:.5f}  noisy_mse={noisy_mse:.5f}  "
              f"val_total={val_total:.4f}  mse_improve=+{improve_pct:.1f}%")
        print(f"         feat_align={val_metrics['val_feat_align']:.5f}  "
              f"inter_feat={val_metrics['val_inter_feat']:.5f}  "
              f"stft_sc={val_metrics['val_stft_sc']:.4f}  "
              f"mel={val_metrics['val_mel_loss']:.4f}")
        print(f"  LR={current_lr:.3e}")

        # best_model_val_total.pt
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
            print(f"  ★ New best val_total: {best_val_total:.4f} → saved")

        # best_model.pt (val_wav_mse)
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
            print(f"  ★ New best val_mse: {best_val_mse:.5f} → saved")

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

    print(f"\n{'='*70}")
    print(f"Training complete. Best val_total: {best_val_total:.4f}")
    print(f"Output: {exp_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

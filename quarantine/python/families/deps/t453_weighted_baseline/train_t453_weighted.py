"""
exp_0217: T453 Token-Aware Curriculum Weighting 訓練腳本

基於 families/deps/encoder_aug/train_augmented.py 修改，核心變更：
1. 使用 T453WeightedSampler 取代 SNR CurriculumSampler
2. 每個 epoch 動態更新採樣權重（高 T453 樣本初期降權）
3. 其餘（LoRA-64、資料增強、loss 設定）與 exp_0216 相同

科學目標：
- T453 加權採樣能否提升 Token Diversity？
  目標: Entropy > 9.3 (vs exp_0216 的 9.21)
  目標: Top-10 mass < 13% (vs exp_0216 的 13.98%)
- 是否影響 Feature MSE 收斂？

對照基線：
- exp_0216 (Aug+LoRA-64): best val MSE = 0.0375, Entropy=9.21, Top10=13.98%
- Teacher ceiling: PESQ=1.345, STOI=0.393

執行：
    # Short-run (1000 steps) - 驗證採樣器是否正常
    python families/deps/t453_weighted_baseline/train_t453_weighted.py --mode step --steps 1000

    # Long-run (300 epochs) - 完整實驗
    python families/deps/t453_weighted_baseline/train_t453_weighted.py --mode epoch --epochs 300

    # 自訂 T453 設定
    python families/deps/t453_weighted_baseline/train_t453_weighted.py --mode epoch --epochs 300 \\
        --t453_min_weight 0.2 --t453_ramp_epochs 150
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
from typing import Dict, List, Optional
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from families.deps.encoder_vq_core.models_single_vq import TeacherStudentSingleVQ
from families.compat_legacy.intermediate_stack.train_v6 import (
    IntermediateSupervisionLossV6,
    verify_model_state,
)
from families.deps.t453_weighted_baseline.data_t453_weighted import (
    create_t453_weighted_dataloaders,
    make_train_loader,
)


# ============================================================
# Utility (來自 train_augmented.py)
# ============================================================

class _TeeIO:
    """同時輸出到 stdout 和 log file。"""

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
    """設定固定隨機種子以確保可重現性。"""
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


def masked_mse(student: torch.Tensor, teacher: torch.Tensor,
               lengths: torch.Tensor = None) -> torch.Tensor:
    if lengths is None:
        return F.mse_loss(student, teacher)

    hop = 320  # 24kHz / 75fps
    T = student.shape[-1]
    frame_lens = (lengths + hop - 1) // hop
    frame_lens = torch.clamp(frame_lens, min=0, max=T)

    frame_idx = torch.arange(T, device=student.device).unsqueeze(0)
    mask = frame_idx < frame_lens.unsqueeze(1)
    mask = mask.unsqueeze(1).to(student.dtype)

    sq = (student - teacher) ** 2 * mask
    denom = mask.sum() * student.shape[1]
    return sq.sum() / denom.clamp(min=1.0)


def get_lora_vq_state_dict(model):
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state[name] = param.data.clone()

    return {
        'lora_state': lora_state,
        'vq_state_dict': model.vq.state_dict(),
    }


# ============================================================
# Training & Evaluation
# ============================================================

@torch.no_grad()
def evaluate_single_vq(model, dataloader, inter_loss_fn, device, config,
                       max_batches: int = 30) -> Dict:
    model.eval()

    all_codes = []
    all_teacher_codes = []
    feature_distances = []
    loss_metrics = {
        'val_total_loss': 0, 'val_loss_quant': 0,
        'val_loss_inter': 0, 'val_loss_commit': 0,
    }

    codebook_size = model.vq_codebook_size
    teacher_codebook_size = int(
        model.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook.shape[0]
    )

    n_batches = 0
    hop = 320

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch.get('lengths')
        if lengths is not None:
            lengths = lengths.to(device)

        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        output = model(clean_audio, noisy_audio)

        loss_quant = masked_mse(output['student_quantized'], output['teacher_encoder_out'], lengths)
        loss_inter, _ = inter_loss_fn(
            student_features=output['student_intermediates'],
            teacher_features=output['teacher_intermediates'],
        )
        loss_commit = output['vq_loss_commit']

        loss_metrics['val_loss_quant'] += loss_quant.item()
        loss_metrics['val_loss_inter'] += loss_inter.item() if isinstance(loss_inter, torch.Tensor) else loss_inter
        loss_metrics['val_loss_commit'] += loss_commit.item()
        loss_metrics['val_total_loss'] += (
            config['lambda_quant'] * loss_quant +
            config['intermediate_weight'] * (loss_inter if isinstance(loss_inter, torch.Tensor) else torch.tensor(loss_inter)) +
            config['beta_commit'] * loss_commit
        ).item()

        student_codes = output['student_codes']  # [1, B, 1, T]
        teacher_codes = output['teacher_codes']  # [1, B, T]

        B = student_codes.shape[1]
        T = student_codes.shape[-1]

        if lengths is not None:
            frame_lens = ((lengths.cpu() + hop - 1) // hop).clamp(min=0, max=T)
            for b in range(B):
                L = int(frame_lens[b].item())
                if L <= 0:
                    continue
                all_codes.append(student_codes[0, b, 0, :L].cpu())
                if teacher_codes.dim() == 3:
                    tc = teacher_codes[0, b, :L]
                elif teacher_codes.dim() == 4:
                    tc = teacher_codes[0, b, 0, :L]
                else:
                    tc = teacher_codes[b, :L]
                all_teacher_codes.append(tc.cpu().flatten())
        else:
            all_codes.append(student_codes[0, :, 0, :].cpu().flatten())
            if teacher_codes.dim() == 3:
                all_teacher_codes.append(teacher_codes[0].cpu().flatten())
            else:
                all_teacher_codes.append(teacher_codes.cpu().flatten())

        student_q = output['student_quantized']
        teacher_e = output['teacher_encoder_out']
        if lengths is not None:
            mse_sum = 0.0
            n_valid = 0
            for b in range(student_q.shape[0]):
                L = int(frame_lens[b].item())
                if L <= 0:
                    continue
                mse_sum += F.mse_loss(student_q[b, :, :L], teacher_e[b, :, :L]).item()
                n_valid += 1
            feature_distances.append(mse_sum / max(1, n_valid))
        else:
            feature_distances.append(F.mse_loss(student_q, teacher_e).item())

        n_batches += 1

    for key in loss_metrics:
        loss_metrics[key] /= max(1, n_batches)

    all_codes_flat = torch.cat(all_codes).flatten().to(torch.int64)
    code_counts = torch.bincount(all_codes_flat, minlength=codebook_size)
    code_probs = code_counts.float() / code_counts.sum()
    nonzero_probs = code_probs[code_probs > 0]
    n_used = int((code_counts > 0).sum().item())
    sorted_probs = torch.sort(code_probs, descending=True).values

    entropy = -(nonzero_probs * torch.log2(nonzero_probs)).sum().item()
    top10_mass = sorted_probs[:min(10, n_used)].sum().item()

    if all_teacher_codes:
        teacher_flat = torch.cat(all_teacher_codes).flatten().to(torch.int64)
        t_counts = torch.bincount(teacher_flat, minlength=teacher_codebook_size)
        t_probs = t_counts.float() / t_counts.sum()
        t_nz = t_probs[t_probs > 0]
        t_entropy = -(t_nz * torch.log2(t_nz)).sum().item()
        t_n_used = int((t_counts > 0).sum().item())
    else:
        t_entropy = 0.0
        t_n_used = 0

    feature_mse = sum(feature_distances) / len(feature_distances) if feature_distances else 0.0
    usage_pct = 100.0 * n_used / codebook_size

    p1_pass = (top10_mass <= 0.95 and n_used >= 82 and feature_mse <= 0.1)
    p2_pass = (entropy >= 5.0 and top10_mass <= 0.5 and n_used >= 410 and feature_mse <= 0.1)
    p3_pass = (entropy > 6.5 and top10_mass < 0.15 and n_used >= 2867)

    model.train()

    return {
        'entropy': entropy,
        'top10_mass': top10_mass,
        'used_codes': n_used,
        'usage_pct': usage_pct,
        'feature_mse': feature_mse,
        'teacher_entropy': t_entropy,
        'teacher_used_codes': t_n_used,
        'p1_pass': bool(p1_pass),
        'p2_pass': bool(p2_pass),
        'p3_pass': bool(p3_pass),
        **loss_metrics,
    }


def save_audio_samples(model, dataloader, device, output_dir, epoch,
                       num_samples=2, split='val'):
    """儲存音檔樣本（noisy / clean / recon）—— 同時保存 train 和 val。"""
    model.eval()
    audio_dir = output_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    data_iter = iter(dataloader)

    for i in range(min(num_samples, len(dataloader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        noisy_audio = batch['noisy_audio'][:1].to(device)
        clean_audio = batch['clean_audio'][:1].to(device)

        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'),
                        noisy_audio.squeeze(0).cpu(), sample_rate)
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'),
                        clean_audio.squeeze(0).cpu(), sample_rate)

        try:
            with torch.no_grad():
                output = model(clean_audio, noisy_audio)
                reconstructed = model.decode(output['student_quantized'])
                if reconstructed.dim() == 3:
                    reconstructed = reconstructed.squeeze(1)
                torchaudio.save(str(audio_dir / f'sample_{i+1}_vq_recon.wav'),
                                reconstructed.cpu(), sample_rate)
        except Exception as e:
            print(f"  Warning: 音檔重建失敗 sample {i+1}: {e}")

    torch.cuda.empty_cache()
    print(f"  Audio saved: {min(num_samples, len(dataloader))} {split} samples → {audio_dir}")


def plot_training_curves(history, output_dir, epoch):
    """繪製訓練曲線（4×3 佈局），包含 T453 weight 追蹤。"""
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f'exp_0217: T453 Weighted Sampling (Epoch {epoch})', fontsize=14)

    epochs = range(1, len(history['train_total_loss']) + 1)

    # Row 1: Losses
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train', alpha=0.7)
    if history.get('val_total_loss'):
        ax.plot(epochs, history['val_total_loss'], 'r-', label='Val', alpha=0.7)
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(epochs, history['train_loss_commit'], 'b-', label='Train Commit', alpha=0.7)
    if history.get('val_loss_commit'):
        ax.plot(epochs, history['val_loss_commit'], 'r-', label='Val Commit', alpha=0.7)
    ax.set_title('Commitment Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(epochs, history['train_loss_quant'], 'b-', label='Train', alpha=0.7)
    if history.get('val_loss_quant'):
        ax.plot(epochs, history['val_loss_quant'], 'r-', label='Val', alpha=0.7)
    ax.set_title('L_quant (post-quant MSE)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Row 2: Collapse Metrics + Intermediate
    ax = axes[1, 0]
    ax.plot(epochs, history['used_codes'], 'darkorange', linewidth=2, label='Used Codes')
    ax.axhline(y=410, color='orange', linestyle=':', alpha=0.5, label='P2 ≥ 410')
    ax.axhline(y=2867, color='red', linestyle=':', alpha=0.5, label='P3 ≥ 2867')
    ax.set_title('Used Codes / 4096')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    if history.get('feature_mse'):
        ax.plot(epochs, history['feature_mse'], 'brown', linewidth=2)
        ax.axhline(y=0.1, color='red', linestyle='--', label='Threshold: ≤0.1')
        ax.axhline(y=0.0375, color='blue', linestyle=':', alpha=0.5, label='exp_0216 best=0.0375')
        ax.set_title('Feature MSE (z_q vs t_e)')
        ax.legend(fontsize=7)
    ax.set_xlabel('Epoch')
    ax.grid(True)

    ax = axes[1, 2]
    ax.plot(epochs, history['train_loss_inter'], 'b-', label='Train Inter', alpha=0.7)
    if history.get('val_loss_inter'):
        ax.plot(epochs, history['val_loss_inter'], 'r-', label='Val Inter', alpha=0.7)
    ax.set_title('Intermediate Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Row 3: Training Dynamics
    ax = axes[2, 0]
    if history.get('train_intermediate_L3_loss'):
        ax.plot(epochs, history['train_intermediate_L3_loss'],
                'b-', label='L3 (w=0.3)', alpha=0.7)
    if history.get('train_intermediate_L4_loss'):
        ax.plot(epochs, history['train_intermediate_L4_loss'],
                'g--', label='L4 (w=0.5)', alpha=0.7)
    if history.get('train_intermediate_L6_loss'):
        ax.plot(epochs, history['train_intermediate_L6_loss'],
                'r-.', label='L6 (w=0.5)', alpha=0.7)
    ax.set_title('Per-Layer Intermediate Loss (Train)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # T453 weight tracking (replaces curriculum_phase)
    ax = axes[2, 1]
    if history.get('t453_high_avg_weight'):
        ax.plot(epochs, history['t453_high_avg_weight'], 'green', linewidth=2,
                label='High-T453 avg weight')
        ax.axhline(y=1.0, color='gray', ls=':', lw=1)
        ax.set_ylim(0, 1.05)
    ax.set_title('T453 High-Concentration Sample Weight')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg sampling weight')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 2]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'green', linewidth=2)
        ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    # Row 4: Codebook Health
    ax = axes[3, 0]
    ax.plot(epochs, history['entropy'], 'darkblue', linewidth=2)
    ax.axhline(y=5.0, color='orange', linestyle=':', alpha=0.7, label='P2 ≥ 5.0')
    ax.axhline(y=6.5, color='red', linestyle='--', alpha=0.7, label='P3 > 6.5')
    ax.axhline(y=9.21, color='purple', linestyle='--', alpha=0.7, label='exp_0216=9.21')
    ax.set_title('Entropy')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    ax = axes[3, 1]
    ax.plot(epochs, history['top10_mass'], 'darkred', linewidth=2)
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='P2 ≤ 0.5')
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='P3 < 0.15')
    ax.axhline(y=0.1398, color='purple', linestyle='--', alpha=0.7, label='exp_0216=13.98%')
    ax.set_title('Top-10 Mass')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    ax = axes[3, 2]
    ax.plot(epochs, history['entropy'], 'b-', linewidth=2, label='Student')
    if history.get('teacher_entropy'):
        ax.plot(epochs, history['teacher_entropy'], 'r--', linewidth=2, label='Teacher')
    ax.set_title('Student vs Teacher Entropy')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch{epoch:03d}.png', dpi=150)
    plt.close()
    print(f"  Loss plot saved: training_curves_epoch{epoch:03d}.png")


def plot_step_metrics(metrics_history: list, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('exp_0217: T453 Weighted Sampling (Short-run)', fontsize=14)

    steps = [m['step'] for m in metrics_history]

    ax = axes[0, 0]
    ax.plot(steps, [m['entropy'] for m in metrics_history], 'b-o', linewidth=2)
    ax.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='P2: ≥5.0')
    ax.axhline(y=6.5, color='green', linestyle='--', alpha=0.7, label='P3: >6.5')
    ax.set_title('Entropy (bits)')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(steps, [m['top10_mass'] for m in metrics_history], 'r-o', linewidth=2)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='P2: ≤0.5')
    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='P3: <0.15')
    ax.set_title('Top-10 Mass')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(steps, [m['used_codes'] for m in metrics_history], 'g-o', linewidth=2)
    ax.axhline(y=410, color='orange', linestyle='--', alpha=0.7, label='P2: ≥410')
    ax.axhline(y=2867, color='green', linestyle='--', alpha=0.7, label='P3: ≥2867')
    ax.set_title('Used Codes / 4096')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(steps, [m['feature_mse'] for m in metrics_history], 'brown', marker='o', linewidth=2)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold: ≤0.1')
    ax.set_title('Feature MSE (z_q vs t_e)')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'metrics_curves_{timestamp}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Metrics plot saved: {plot_path}")


# ============================================================
# Step-based training (short-run)
# ============================================================

def train_step_based(model, train_loader, optimizer, inter_loss_fn, device,
                     config, scaler=None):
    model.train()

    total_steps = config['steps']
    eval_interval = config['eval_interval']
    checkpoint_interval = config.get('checkpoint_interval', eval_interval)
    intermediate_weight = config['intermediate_weight']
    output_dir = Path(config['output_dir'])

    metrics_history = []
    nan_count = 0
    max_nan = 50

    step = 0
    data_iter = iter(train_loader)

    running_metrics = {
        'total_loss': 0, 'loss_quant': 0, 'loss_inter': 0,
        'loss_commit': 0, 'loss_codebook': 0,
    }
    running_count = 0

    pbar = tqdm(total=total_steps, desc="Training")

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch.get('lengths')
        if lengths is not None:
            lengths = lengths.to(device)

        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        if step % config['grad_accum'] == 0:
            optimizer.zero_grad()

        with autocast(enabled=config['use_amp']):
            output = model(clean_audio, noisy_audio)

            loss_quant = masked_mse(
                student=output['student_quantized'],
                teacher=output['teacher_encoder_out'],
                lengths=lengths,
            )

            loss_inter_raw, inter_info = inter_loss_fn(
                student_features=output['student_intermediates'],
                teacher_features=output['teacher_intermediates'],
            )
            loss_inter = loss_inter_raw

            loss_commit = output['vq_loss_commit']
            loss_codebook = output['vq_loss_codebook']

            total_loss = (
                config['lambda_quant'] * loss_quant +
                intermediate_weight * loss_inter +
                config['beta_commit'] * loss_commit +
                config['lambda_codebook'] * loss_codebook
            )
            total_loss = total_loss / config['grad_accum']

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            nan_count += 1
            print(f"  NaN/Inf at step {step}, skipping (count: {nan_count})")
            optimizer.zero_grad()
            if nan_count >= max_nan:
                print(f"  Too many NaN ({nan_count}), aborting!")
                break
            continue

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if (step + 1) % config['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip'],
                )
                scaler.step(optimizer)
                scaler.update()
        else:
            total_loss.backward()
            if (step + 1) % config['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip'],
                )
                optimizer.step()

        running_metrics['total_loss'] += total_loss.item() * config['grad_accum']
        running_metrics['loss_quant'] += loss_quant.item()
        running_metrics['loss_inter'] += loss_inter.item() if isinstance(loss_inter, torch.Tensor) else loss_inter
        running_metrics['loss_commit'] += loss_commit.item()
        running_metrics['loss_codebook'] += loss_codebook.item()
        running_count += 1

        step += 1
        pbar.update(1)
        pbar.set_postfix({
            'loss': f"{total_loss.item() * config['grad_accum']:.4f}",
            'quant': f"{loss_quant.item():.4f}",
            'commit': f"{loss_commit.item():.4f}",
        })

        if step % eval_interval == 0 or step == total_steps:
            avg_train = {k: v / max(1, running_count) for k, v in running_metrics.items()}

            print(f"\n--- Step {step}/{total_steps}: Evaluating ---")
            val_metrics = evaluate_single_vq(
                model, train_loader, inter_loss_fn, device, config,
                max_batches=config.get('eval_max_batches', 30),
            )

            record = {
                'step': step,
                'train_total_loss': avg_train['total_loss'],
                'train_loss_quant': avg_train['loss_quant'],
                'train_loss_inter': avg_train['loss_inter'],
                'train_loss_commit': avg_train['loss_commit'],
                **val_metrics,
            }
            metrics_history.append(record)

            print(f"  Train: loss={avg_train['total_loss']:.4f} "
                  f"quant={avg_train['loss_quant']:.4f} "
                  f"commit={avg_train['loss_commit']:.4f}")
            print(f"  Eval:  entropy={val_metrics['entropy']:.3f} "
                  f"top10={val_metrics['top10_mass']:.4f} "
                  f"used={val_metrics['used_codes']}/{config.get('codebook_size', 4096)} "
                  f"mse={val_metrics['feature_mse']:.4f}")
            print(f"  Gates: P1={'pass' if val_metrics['p1_pass'] else 'fail'} "
                  f"P2={'pass' if val_metrics['p2_pass'] else 'fail'} "
                  f"P3={'pass' if val_metrics['p3_pass'] else 'fail'}")

            running_metrics = {k: 0 for k in running_metrics}
            running_count = 0

            with open(output_dir / 'metrics_history.json', 'w') as f:
                json.dump(metrics_history, f, indent=2)

            if step % checkpoint_interval == 0:
                ckpt_dir = output_dir / 'checkpoints'
                ckpt_dir.mkdir(exist_ok=True)
                ckpt_path = ckpt_dir / f'checkpoint_step{step:06d}.pt'
                torch.save({
                    'step': step,
                    **get_lora_vq_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': config,
                }, ckpt_path)
                print(f"  Checkpoint: {ckpt_path.name}")

            model.train()

    pbar.close()

    if metrics_history:
        plot_step_metrics(metrics_history, output_dir)

    return metrics_history


# ============================================================
# Epoch-based training
# ============================================================

def train_epoch(model, dataloader, optimizer, inter_loss_fn, device, epoch,
                config, scaler=None):
    model.train()
    verify_model_state(model, f"Epoch {epoch} start")

    intermediate_weight = config['intermediate_weight']
    nan_count = 0
    max_nan_per_epoch = 10

    metrics = {
        'total_loss': 0, 'loss_quant': 0,
        'loss_inter': 0, 'loss_commit': 0, 'loss_codebook': 0,
        'intermediate_weight': intermediate_weight,
        'intermediate_L3_loss': 0, 'intermediate_L4_loss': 0,
        'intermediate_L6_loss': 0,
        'nan_batches': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch.get('lengths')
        if lengths is not None:
            lengths = lengths.to(device)

        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        if batch_idx % config['grad_accum'] == 0:
            optimizer.zero_grad()

        with autocast(enabled=config['use_amp']):
            output = model(clean_audio, noisy_audio)

            loss_quant = masked_mse(
                student=output['student_quantized'],
                teacher=output['teacher_encoder_out'],
                lengths=lengths,
            )

            loss_inter_raw, inter_info = inter_loss_fn(
                student_features=output['student_intermediates'],
                teacher_features=output['teacher_intermediates'],
            )
            loss_inter = loss_inter_raw

            loss_commit = output['vq_loss_commit']
            loss_codebook = output['vq_loss_codebook']

            total_loss = (
                config['lambda_quant'] * loss_quant +
                intermediate_weight * loss_inter +
                config['beta_commit'] * loss_commit +
                config['lambda_codebook'] * loss_codebook
            )
            total_loss = total_loss / config['grad_accum']

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            nan_count += 1
            metrics['nan_batches'] = nan_count
            print(f"  NaN/Inf at batch {batch_idx}, skipping (count: {nan_count})")
            optimizer.zero_grad()
            if nan_count >= max_nan_per_epoch:
                print(f"  Too many NaN batches ({nan_count})")
            continue

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip'],
                )
                scaler.step(optimizer)
                scaler.update()
        else:
            total_loss.backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip'],
                )
                optimizer.step()

        metrics['total_loss'] += total_loss.item() * config['grad_accum']
        metrics['loss_quant'] += loss_quant.item()
        metrics['loss_inter'] += loss_inter.item() if isinstance(loss_inter, torch.Tensor) else loss_inter
        metrics['loss_commit'] += loss_commit.item()
        metrics['loss_codebook'] += loss_codebook.item()

        for lkey in ['intermediate_L3_loss', 'intermediate_L4_loss', 'intermediate_L6_loss']:
            if lkey in inter_info:
                metrics[lkey] += inter_info[lkey]

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{total_loss.item() * config['grad_accum']:.4f}",
            'quant': f"{loss_quant.item():.4f}",
            'commit': f"{loss_commit.item():.4f}",
        })

    for key in metrics:
        if key not in ('intermediate_weight', 'nan_batches'):
            metrics[key] /= max(1, n_batches)

    return metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='exp_0217: T453 Token-Aware Curriculum Weighting'
    )

    # Training mode
    parser.add_argument('--mode', type=str, default='step',
                        choices=['step', 'epoch'],
                        help='訓練模式: step (short-run) 或 epoch (long-run)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Step-based 模式的總步數')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Epoch-based 模式的總 epochs')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄')

    # Training basics
    parser.add_argument('--seed', type=int, default=42, help='固定隨機種子')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=2, help='梯度累積步數')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小學習率')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--use_amp', action='store_true', default=True, help='混合精度')
    parser.add_argument('--device', type=str, default='cuda:0', help='裝置')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=128, help='LoRA alpha')

    # Single VQ + EMA
    parser.add_argument('--vq_ema_decay', type=float, default=0.99, help='EMA decay')
    parser.add_argument('--vq_ema_threshold', type=int, default=2,
                        help='Dead-code reset threshold')
    parser.add_argument('--vq_ema_usage_penalty', type=float, default=0.0)

    # Loss weights
    parser.add_argument('--lambda_quant', type=float, default=1.0)
    parser.add_argument('--lambda_codebook', type=float, default=0.0)
    parser.add_argument('--beta_commit', type=float, default=1.0)
    parser.add_argument('--intermediate_weight', type=float, default=0.03)

    # Intermediate layer weights
    parser.add_argument('--intermediate_L3_weight', type=float, default=0.3)
    parser.add_argument('--intermediate_L4_weight', type=float, default=0.5)
    parser.add_argument('--intermediate_L6_weight', type=float, default=0.5)

    # T453 Weighting (核心新增)
    parser.add_argument('--t453_min_weight', type=float, default=0.2,
                        help='Epoch 0 時高 T453 樣本的最低採樣權重')
    parser.add_argument('--t453_ramp_epochs', type=int, default=150,
                        help='T453 weight 從 min_weight 升到 1.0 所需 epoch 數')

    # Data Augmentation
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
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Step-based: 每 N steps 評估')
    parser.add_argument('--checkpoint_interval', type=int, default=200)
    parser.add_argument('--eval_max_batches', type=int, default=30)
    parser.add_argument('--save_checkpoint_every', type=int, default=10,
                        help='Epoch-based: 每 N epochs 存 checkpoint')
    parser.add_argument('--save_audio_interval', type=int, default=50,
                        help='Epoch-based: 每 N epochs 存 train+val 音檔 + 繪圖')

    args = parser.parse_args()

    # ===== Setup =====
    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(f'families/deps/t453_weighted_baseline/runs/t453_weighted_{args.mode}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    config = vars(args)
    config['timestamp'] = timestamp
    config['output_dir'] = str(exp_dir)
    config['intermediate_indices'] = [3, 4, 6]
    config['codebook_size'] = 4096
    config['experiment'] = 'exp_0217_t453_weighted'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0217: T453 Token-Aware Curriculum Weighting")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    if args.mode == 'step':
        print(f"Steps: {args.steps}, Eval every {args.eval_interval} steps")
    else:
        print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed} (fixed)")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"T453 Weighting: min_weight={args.t453_min_weight}, ramp_epochs={args.t453_ramp_epochs}")
    print(f"Loss: lambda_quant={args.lambda_quant}, beta_commit={args.beta_commit}, "
          f"inter={args.intermediate_weight}")
    print(f"Audio: saved every {args.save_audio_interval} epochs (train + val)")
    print(f"Output: {exp_dir}")
    if log_path:
        print(f"Log: {log_path}")
    print("=" * 70)

    # ===== CUDA =====
    device = torch.device(args.device)
    cuda_preinit(device)

    # ===== Data (T453 weighted, with augmentation) =====
    print("\nLoading data with T453 weighting...")
    train_dataset, val_loader, t453_sampler = create_t453_weighted_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=2,
        total_epochs=args.epochs if args.mode == 'epoch' else 300,
        t453_min_weight=args.t453_min_weight,
        t453_ramp_epochs=args.t453_ramp_epochs,
        # 增強參數 (與 exp_0216 相同)
        snr_remix_prob=args.snr_remix_prob,
        snr_remix_range=(args.snr_remix_min, args.snr_remix_max),
        random_gain_prob=args.random_gain_prob,
        random_gain_db=args.random_gain_db,
        random_crop_prob=args.random_crop_prob,
        random_crop_min_ratio=args.random_crop_min_ratio,
        time_stretch_prob=args.time_stretch_prob,
        time_stretch_range=(args.time_stretch_min, args.time_stretch_max),
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # ===== Model =====
    print("\nBuilding model...")
    model = TeacherStudentSingleVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6],
        device=device,
        vq_ema_decay=args.vq_ema_decay,
        vq_ema_threshold=args.vq_ema_threshold,
        vq_ema_usage_penalty=args.vq_ema_usage_penalty,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params_count:,} ({100*trainable_params_count/total_params:.1f}%)")

    inter_loss_fn = IntermediateSupervisionLossV6(
        layer_weights={
            3: args.intermediate_L3_weight,
            4: args.intermediate_L4_weight,
            6: args.intermediate_L6_weight,
        },
    )

    # ===== Optimizer =====
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler() if args.use_amp else None

    # ===== Training =====
    if args.mode == 'step':
        print(f"\nStarting step-based training ({args.steps} steps)...")
        # For step-based, use epoch 0 loader (fixed T453 weights)
        train_loader = make_train_loader(
            train_dataset, t453_sampler, epoch=0,
            batch_size=args.batch_size, num_workers=2,
        )
        metrics_history = train_step_based(
            model, train_loader, optimizer, inter_loss_fn, device,
            config=config, scaler=scaler,
        )

        if metrics_history:
            final = metrics_history[-1]
            summary = {
                'config': config,
                'final_metrics': final,
                'acceptance': {
                    'P1_pass': final.get('p1_pass', False),
                    'P2_pass': final.get('p2_pass', False),
                    'P3_pass': final.get('p3_pass', False),
                },
            }
            with open(exp_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            print("\n" + "=" * 70)
            print("Short-run done!")
            print("=" * 70)
            print(f"  Entropy:     {final['entropy']:.3f}")
            print(f"  Top-10:      {final['top10_mass']:.4f}")
            print(f"  Used codes:  {final['used_codes']}/4096 ({final['usage_pct']:.1f}%)")
            print(f"  Feature MSE: {final['feature_mse']:.4f}")
            print("=" * 70)

    else:
        # Epoch-based training (long-run)
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return epoch / args.warmup_epochs
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return max(args.min_lr / args.learning_rate,
                       0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        history = {
            'train_total_loss': [], 'train_loss_quant': [],
            'train_loss_inter': [], 'train_loss_commit': [],
            'val_total_loss': [], 'val_loss_quant': [],
            'val_loss_inter': [], 'val_loss_commit': [],
            'entropy': [], 'top10_mass': [],
            'used_codes': [], 'feature_mse': [],
            'lr': [], 'p2_pass': [], 'p3_pass': [],
            't453_high_avg_weight': [],   # T453 weight tracking
            'train_intermediate_L3_loss': [],
            'train_intermediate_L4_loss': [],
            'train_intermediate_L6_loss': [],
            'teacher_entropy': [],
        }

        best_val_loss = float('inf')
        best_val_mse = float('inf')

        print(f"\nStarting epoch-based training ({args.epochs} epochs)...")
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # ── T453 weighted loader: recreate each epoch with updated weights ──
            train_loader = make_train_loader(
                train_dataset, t453_sampler, epoch - 1,  # 0-indexed
                batch_size=args.batch_size, num_workers=2,
            )

            # Track T453 weight for this epoch
            weights = t453_sampler._compute_weights(epoch - 1)
            high_mask = t453_sampler.t453_ratios > 0.3
            if high_mask.any():
                t453_high_avg = float(weights[high_mask].mean())
            else:
                t453_high_avg = 1.0

            train_metrics = train_epoch(
                model, train_loader, optimizer, inter_loss_fn, device, epoch,
                config=config, scaler=scaler,
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            val_metrics = evaluate_single_vq(
                model, val_loader, inter_loss_fn, device, config,
                max_batches=args.eval_max_batches,
            )

            epoch_time = time.time() - epoch_start

            history['train_total_loss'].append(train_metrics['total_loss'])
            history['train_loss_quant'].append(train_metrics['loss_quant'])
            history['train_loss_inter'].append(train_metrics['loss_inter'])
            history['train_loss_commit'].append(train_metrics['loss_commit'])
            history['val_total_loss'].append(val_metrics['val_total_loss'])
            history['val_loss_quant'].append(val_metrics.get('val_loss_quant', 0))
            history['val_loss_inter'].append(val_metrics.get('val_loss_inter', 0))
            history['val_loss_commit'].append(val_metrics.get('val_loss_commit', 0))
            history['entropy'].append(val_metrics['entropy'])
            history['top10_mass'].append(val_metrics['top10_mass'])
            history['used_codes'].append(val_metrics['used_codes'])
            history['feature_mse'].append(val_metrics['feature_mse'])
            history['lr'].append(current_lr)
            history['p2_pass'].append(val_metrics['p2_pass'])
            history['p3_pass'].append(val_metrics['p3_pass'])
            history['t453_high_avg_weight'].append(t453_high_avg)
            history['teacher_entropy'].append(val_metrics.get('teacher_entropy', 0))
            history['train_intermediate_L3_loss'].append(
                train_metrics.get('intermediate_L3_loss', 0))
            history['train_intermediate_L4_loss'].append(
                train_metrics.get('intermediate_L4_loss', 0))
            history['train_intermediate_L6_loss'].append(
                train_metrics.get('intermediate_L6_loss', 0))

            print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train: loss={train_metrics['total_loss']:.4f} "
                  f"quant={train_metrics['loss_quant']:.4f} "
                  f"commit={train_metrics['loss_commit']:.4f}")
            print(f"  Eval:  entropy={val_metrics['entropy']:.3f} "
                  f"top10={val_metrics['top10_mass']:.4f} "
                  f"used={val_metrics['used_codes']}/4096 "
                  f"mse={val_metrics['feature_mse']:.4f}")
            print(f"  T453:  high-sample avg_weight={t453_high_avg:.3f} "
                  f"(ramped={t453_sampler.is_fully_ramped(epoch-1)})")
            print(f"  P2={'pass' if val_metrics['p2_pass'] else 'fail'} "
                  f"P3={'pass' if val_metrics['p3_pass'] else 'fail'} "
                  f"LR={current_lr:.2e}")

            if val_metrics['feature_mse'] < best_val_mse:
                best_val_mse = val_metrics['feature_mse']
                print(f"  New best val MSE: {best_val_mse:.4f} (epoch {epoch})")

            if val_metrics['val_total_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'vq_state_dict': model.vq.state_dict(),
                    'metrics': val_metrics,
                    'config': config,
                }, exp_dir / 'best_model.pt')

            if epoch % args.save_checkpoint_every == 0:
                ckpt_dir = exp_dir / 'checkpoints'
                ckpt_dir.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    **get_lora_vq_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': val_metrics,
                    'config': config,
                }, ckpt_dir / f'checkpoint_epoch{epoch:03d}.pt')

            with open(exp_dir / 'metrics_history.json', 'w') as f:
                json.dump(history, f, indent=2)

            # Save audio samples (train + val) + loss plots
            if epoch % args.save_audio_interval == 0 or epoch == 1:
                try:
                    save_audio_samples(model, val_loader, device, exp_dir,
                                       epoch, num_samples=2, split='val')
                    save_audio_samples(model, train_loader, device, exp_dir,
                                       epoch, num_samples=2, split='train')
                except Exception as e:
                    print(f"  Warning: Audio save failed: {e}")
                try:
                    plot_training_curves(history, exp_dir, epoch)
                except Exception as e:
                    print(f"  Warning: Plot failed: {e}")

            gc.collect()
            torch.cuda.empty_cache()

        # Final
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'vq_state_dict': model.vq.state_dict(),
            'config': config,
        }, exp_dir / 'final_model.pt')

        try:
            plot_training_curves(history, exp_dir, args.epochs)
        except Exception as e:
            print(f"  Warning: Final plot failed: {e}")

        summary = {
            'experiment': 'exp_0217_t453_weighted',
            'mode': 'epoch',
            'total_epochs': args.epochs,
            'seed': args.seed,
            'config': config,
            'final_metrics': {
                'train_total_loss': history['train_total_loss'][-1],
                'val_total_loss': history['val_total_loss'][-1],
                'entropy': history['entropy'][-1],
                'top10_mass': history['top10_mass'][-1],
                'used_codes': history['used_codes'][-1],
                'feature_mse': history['feature_mse'][-1],
                'p2_pass': history['p2_pass'][-1],
                'p3_pass': history['p3_pass'][-1],
            },
            'best_val_loss': best_val_loss,
            'best_val_mse': best_val_mse,
            'baseline_reference': {
                'exp_0216_best_val_mse': 0.0375,
                'exp_0216_entropy': 9.21,
                'exp_0216_top10_mass': 0.1398,
                'teacher_ceiling_pesq': 1.345,
                'teacher_ceiling_stoi': 0.393,
            },
            'changes_vs_exp_0216': {
                'sampling': 'SNR Curriculum → T453 Weighted Sampling',
                't453_min_weight': args.t453_min_weight,
                't453_ramp_epochs': args.t453_ramp_epochs,
                'seed': f'fixed={args.seed}',
            },
        }
        with open(exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {exp_dir / 'summary.json'}")

    print(f"\nTraining done! Results at {exp_dir}")


if __name__ == '__main__':
    main()

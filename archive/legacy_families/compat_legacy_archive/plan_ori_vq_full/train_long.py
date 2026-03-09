"""
exp_0206: Long-Term RVQ Training (300 Epochs)

結合 exp_k_v6 的 epoch-based 訓練框架 + Phase 3-2 Exp 6c 的 RVQ + EMA + Dead-Code Reset。

設計重點:
1. 架構: TeacherStudentRVQ (LoRA + 4-layer RVQ, EMA codebook, dead-code reset)
2. 損失: L_quant(post-quant) + L_inter(V6 warmdown) + L_commit(encoder commitment)
3. 訓練: 300 epochs, curriculum learning, AMP, gradient accumulation
4. 評估: Phase 3-2 P1/P2/P3 驗收 + exp_k_v6 style token accuracy
5. 儲存: 每 10 epochs checkpoint, 每 50 epochs audio samples

執行:
    bash families/compat_legacy/plan_ori_vq/run.sh [GPU_ID]
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
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from families.compat_legacy.residual_vq_stack.phase3.residual_vq.models_rvq import TeacherStudentRVQ
from families.compat_legacy.intermediate_stack.train_v6 import (
    IntermediateSupervisionLossV6,
    compute_dynamic_intermediate_weight,
    verify_model_state,
)
from families.compat_legacy.curriculum_data.data_curriculum import (
    create_curriculum_dataloaders,
    CurriculumSampler,
)


# ============================================================
# Utility
# ============================================================

class _TeeIO:
    """同時輸出到 stdout 和 log file"""

    def __init__(self, *streams):
        """初始化多重輸出流

        Args:
            *streams: 要同時寫入的多個輸出流
        """
        self._streams = streams

    def write(self, data):
        """寫入資料到所有流

        Args:
            data: 要寫入的字串資料
        """
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        """刷新所有流"""
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        """回傳是否為終端設備

        Returns:
            永遠回傳 False（因為有 file logging）
        """
        return False


def setup_logging(output_dir: Path) -> Path:
    """設定日誌輸出到檔案

    Args:
        output_dir: 輸出目錄路徑

    Returns:
        log 檔案路徑，若建立失敗則回傳 None
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
    """設定隨機種子以確保可重現性

    Args:
        seed: 隨機種子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_preinit(device: torch.device, retries: int = 10, sleep_s: float = 2.0):
    """預先初始化 CUDA 裝置（避免延遲啟動問題）

    Args:
        device: CUDA 裝置
        retries: 最大重試次數
        sleep_s: 每次重試間隔秒數
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


def masked_mse(student: torch.Tensor, teacher: torch.Tensor,
               lengths: torch.Tensor = None) -> torch.Tensor:
    """計算帶 mask 的 MSE（僅在有效 frame 上計算）

    Args:
        student: 學生模型輸出 [B, C, T]
        teacher: 教師模型輸出 [B, C, T]
        lengths: 每個樣本的音訊長度（24kHz sample 數）

    Returns:
        masked MSE loss
    """
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


def get_lora_rvq_state_dict(model):
    """提取 LoRA + RVQ 參數（用於 checkpoint 節省空間）

    Args:
        model: TeacherStudentRVQ 模型

    Returns:
        包含 LoRA 參數和 RVQ 狀態的字典
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state[name] = param.data.clone()

    return {
        'lora_state': lora_state,
        'rvq_state_dict': model.rvq.state_dict(),
    }


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, dataloader, optimizer, inter_loss_fn, device, epoch,
                config, scaler=None, curriculum_sampler=None):
    """訓練一個 epoch

    Args:
        model: TeacherStudentRVQ 模型
        dataloader: 訓練資料載入器
        optimizer: 優化器
        inter_loss_fn: 中間層監督損失函數
        device: 計算裝置
        epoch: 當前 epoch 數
        config: 訓練配置字典
        scaler: AMP GradScaler（若使用混合精度）
        curriculum_sampler: 課程學習取樣器（用於 advance_phase）

    Returns:
        包含各項損失和指標的字典
    """
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    # 計算動態中間層權重
    intermediate_weight = compute_dynamic_intermediate_weight(
        epoch=epoch,
        curriculum_epochs=config['curriculum_epochs'],
        base_weight=config['intermediate_weight'],
        min_weight=config['intermediate_weight_min'],
        warmdown_epochs=config['warmdown_epochs'],
    )

    # 計算 usage penalty（隨 epoch 線性遞增，可選）
    total_steps_per_epoch = len(dataloader)
    global_step = (epoch - 1) * total_steps_per_epoch

    metrics = {
        'total_loss': 0, 'loss_quant': 0, 'loss_pre': 0,
        'loss_inter': 0, 'loss_commit': 0, 'loss_codebook': 0,
        'intermediate_weight': intermediate_weight,
        'intermediate_L3_loss': 0, 'intermediate_L4_loss': 0,
        'intermediate_L6_loss': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch.get('lengths')
        if lengths is not None:
            lengths = lengths.to(device)

        # 確保維度 [B, 1, T]
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        if batch_idx % config['grad_accum'] == 0:
            optimizer.zero_grad()

        with autocast(enabled=config['use_amp']):
            output = model(clean_audio, noisy_audio)

            # Phase 3-2: Post-quant alignment (主損失)
            loss_quant = masked_mse(
                student=output['student_quantized'],
                teacher=output['teacher_encoder_out'],
                lengths=lengths,
            )

            # Pre-quant alignment (disabled, λ=0)
            loss_pre = masked_mse(
                student=output['student_encoder_out'],
                teacher=output['teacher_encoder_out'],
                lengths=lengths,
            )

            # Intermediate supervision (V6 cosine loss)
            loss_inter_raw, inter_info = inter_loss_fn(
                student_features=output['student_intermediates'],
                teacher_features=output['teacher_intermediates'],
            )
            loss_inter = loss_inter_raw

            # RVQ commitment & codebook losses
            loss_commit = output['rvq_loss_commit']
            loss_codebook = output['rvq_loss_codebook']

            # 總損失
            total_loss = (
                config['lambda_quant'] * loss_quant +
                config['lambda_pre'] * loss_pre +
                intermediate_weight * loss_inter +
                config['beta_commit'] * loss_commit +
                config['lambda_codebook'] * loss_codebook
            )

            total_loss = total_loss / config['grad_accum']

        # Backward
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip']
                )
                scaler.step(optimizer)
                scaler.update()
        else:
            total_loss.backward()
            if (batch_idx + 1) % config['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config['grad_clip']
                )
                optimizer.step()

        # 記錄指標
        metrics['total_loss'] += total_loss.item() * config['grad_accum']
        metrics['loss_quant'] += loss_quant.item()
        metrics['loss_pre'] += loss_pre.item()
        metrics['loss_inter'] += loss_inter.item() if isinstance(loss_inter, torch.Tensor) else loss_inter
        metrics['loss_commit'] += loss_commit.item()
        metrics['loss_codebook'] += loss_codebook.item()

        # 記錄 per-layer intermediate loss
        for lkey in ['intermediate_L3_loss', 'intermediate_L4_loss', 'intermediate_L6_loss']:
            if lkey in inter_info:
                metrics[lkey] += inter_info[lkey]

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{total_loss.item() * config['grad_accum']:.4f}",
            'quant': f"{loss_quant.item():.4f}",
            'inter': f"{loss_inter.item() if isinstance(loss_inter, torch.Tensor) else loss_inter:.4f}",
            'iw': f"{intermediate_weight:.3f}",
        })

    # 平均
    for key in metrics:
        if key != 'intermediate_weight':
            metrics[key] /= max(1, n_batches)

    return metrics


@torch.no_grad()
def evaluate_epoch(model, val_loader, inter_loss_fn, device, config,
                   max_batches: int = None):
    """評估一個 epoch（結合 Phase 3-2 collapse metrics + val loss）

    Args:
        model: TeacherStudentRVQ 模型
        val_loader: 驗證資料載入器
        inter_loss_fn: 中間層監督損失函數
        device: 計算裝置
        config: 訓練配置字典
        max_batches: 最大評估批次數

    Returns:
        包含所有評估指標的字典
    """
    model.eval()

    # Val loss metrics
    loss_metrics = {
        'val_total_loss': 0, 'val_loss_quant': 0, 'val_loss_pre': 0,
        'val_loss_inter': 0, 'val_loss_commit': 0, 'val_loss_codebook': 0,
    }

    # Collapse metrics
    all_layer0_codes = []
    all_layer_codes_list = []
    all_teacher_codes = []
    feature_distances = []

    codebook_size = model.rvq_codebook_size
    teacher_codebook_size = int(
        model.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook.shape[0]
    )

    n_batches = 0

    for i, batch in enumerate(val_loader):
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

        # Val losses
        loss_quant = masked_mse(output['student_quantized'], output['teacher_encoder_out'], lengths)
        loss_pre = masked_mse(output['student_encoder_out'], output['teacher_encoder_out'], lengths)
        loss_inter, _ = inter_loss_fn(
            student_features=output['student_intermediates'],
            teacher_features=output['teacher_intermediates'],
        )
        loss_commit = output['rvq_loss_commit']
        loss_codebook = output['rvq_loss_codebook']

        loss_metrics['val_total_loss'] += (
            config['lambda_quant'] * loss_quant +
            config['lambda_pre'] * loss_pre +
            config['intermediate_weight'] * (loss_inter if isinstance(loss_inter, torch.Tensor) else torch.tensor(loss_inter)) +
            config['beta_commit'] * loss_commit +
            config['lambda_codebook'] * loss_codebook
        ).item()
        loss_metrics['val_loss_quant'] += loss_quant.item()
        loss_metrics['val_loss_pre'] += loss_pre.item()
        loss_metrics['val_loss_inter'] += loss_inter.item() if isinstance(loss_inter, torch.Tensor) else loss_inter
        loss_metrics['val_loss_commit'] += loss_commit.item()
        loss_metrics['val_loss_codebook'] += loss_codebook.item()

        # Collapse metrics collection
        all_layer_codes = output['all_layer_codes'].cpu()
        teacher_codes = output['teacher_codes'].cpu()
        n_layers, batch_size, time_len = all_layer_codes.shape

        if teacher_codes.dim() == 3 and teacher_codes.shape[0] != batch_size:
            teacher_codes = teacher_codes.permute(1, 0, 2).contiguous()

        hop = 320
        if lengths is not None:
            frame_lens = ((lengths.cpu() + hop - 1) // hop).clamp(min=0, max=time_len)
            for b in range(batch_size):
                L = int(frame_lens[b].item())
                if L <= 0:
                    continue
                all_layer0_codes.append(all_layer_codes[0, b, :L])
                all_layer_codes_list.append(all_layer_codes[:, b, :L])
                tc = teacher_codes[b]
                Lt = min(L, int(tc.shape[-1]))
                if tc.dim() == 1:
                    all_teacher_codes.append(tc[:Lt].reshape(-1))
                elif tc.dim() == 2:
                    all_teacher_codes.append(tc[:, :Lt].reshape(-1))
                else:
                    all_teacher_codes.append(tc.reshape(-1))
        else:
            all_layer0_codes.append(all_layer_codes[0].reshape(-1))
            all_layer_codes_list.append(all_layer_codes.reshape(n_layers, -1))
            all_teacher_codes.append(teacher_codes.reshape(-1))

        # Feature MSE
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

    # 平均 losses
    for key in loss_metrics:
        loss_metrics[key] /= max(1, n_batches)

    # ===== Collapse Metrics =====
    all_layer0_codes = torch.cat(all_layer0_codes).flatten()
    code_counts = torch.bincount(all_layer0_codes, minlength=codebook_size)
    code_probs = code_counts.float() / code_counts.sum()
    nonzero_probs = code_probs[code_probs > 0]
    n_used = int((code_counts > 0).sum().item())
    sorted_probs = torch.sort(code_probs, descending=True).values

    entropy = -(nonzero_probs * torch.log2(nonzero_probs)).sum().item()

    def _topk_mass(sorted_p, k, n):
        """計算 top-k 佔比"""
        if n <= 0:
            return 0.0
        k_eff = min(int(k), int(n))
        return sorted_p[:k_eff].sum().item()

    top10_mass = _topk_mass(sorted_probs, 10, n_used)

    # Teacher distribution
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

    # Joint diversity
    all_layer_codes_flat = torch.cat(all_layer_codes_list, dim=1)
    joint_codes = all_layer_codes_flat.t().contiguous()
    unique_joint = torch.unique(joint_codes, dim=0).shape[0]
    joint_diversity = float(unique_joint) / float(joint_codes.shape[0])

    # Feature MSE
    feature_mse = sum(feature_distances) / len(feature_distances)

    # Per-layer usage
    layer_usage = {}
    for li in range(all_layer_codes_flat.shape[0]):
        lc = all_layer_codes_flat[li].flatten()
        lu = torch.bincount(lc, minlength=codebook_size)
        lu_used = int((lu > 0).sum().item())
        lu_probs = lu.float() / lu.sum()
        lu_nz = lu_probs[lu_probs > 0]
        lu_ent = -(lu_nz * torch.log2(lu_nz)).sum().item()
        layer_usage[f'layer_{li}_used'] = lu_used
        layer_usage[f'layer_{li}_entropy'] = lu_ent
        layer_usage[f'layer_{li}_usage_pct'] = 100.0 * lu_used / codebook_size

    # Acceptance gates
    K = codebook_size
    p2_pass = (
        entropy >= 5.0 and
        top10_mass <= 0.5 and
        n_used >= int(0.10 * K) and
        joint_diversity >= 0.30 and
        feature_mse <= 0.1
    )
    p3_pass = (
        entropy > 6.5 and
        top10_mass < 0.15 and
        joint_diversity > 0.7 and
        feature_mse < 0.1
    )

    model.train()

    collapse_metrics = {
        # Layer 0
        'layer0_entropy': entropy,
        'layer0_top10_mass': top10_mass,
        'layer0_used_codes': n_used,
        'layer0_usage_pct': 100.0 * n_used / codebook_size,
        # Teacher
        'teacher_entropy': t_entropy,
        'teacher_used_codes': t_n_used,
        # Joint
        'joint_diversity': joint_diversity,
        'joint_unique_codes': unique_joint,
        # Feature
        'feature_mse': feature_mse,
        # Acceptance
        'p2_pass': bool(p2_pass),
        'p3_pass': bool(p3_pass),
    }
    collapse_metrics.update(layer_usage)
    collapse_metrics.update(loss_metrics)

    return collapse_metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, output_dir, epoch,
                       num_samples=2, split='val'):
    """保存音檔樣本（使用 Teacher decoder 重建 RVQ quantized vectors）

    Args:
        model: TeacherStudentRVQ 模型
        dataloader: 資料載入器
        device: 計算裝置
        output_dir: 輸出目錄
        epoch: 當前 epoch
        num_samples: 要儲存的樣本數
        split: 'val' 或 'train'
    """
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

        # 儲存 noisy / clean 原始音檔
        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'),
                        noisy_audio.squeeze(0).cpu(), sample_rate)
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'),
                        clean_audio.squeeze(0).cpu(), sample_rate)

        try:
            output = model(clean_audio, noisy_audio)
            reconstructed = model.decode(output['student_quantized'])
            if reconstructed.dim() == 3:
                reconstructed = reconstructed.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_rvq_recon.wav'),
                            reconstructed.cpu(), sample_rate)
        except Exception as e:
            print(f"  Warning: 音檔重建失敗 sample {i+1}: {e}")

    torch.cuda.empty_cache()
    print(f"  已儲存 {min(num_samples, len(dataloader))} 個 {split} 音檔樣本")


def plot_training_curves(history, output_dir, epoch):
    """繪製訓練曲線（3×3 佈局，結合 v6 baseline 風格 + Phase 3-2 collapse metrics）

    Args:
        history: 訓練歷史字典
        output_dir: 輸出目錄
        epoch: 當前 epoch（用於圖表標題）
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'exp_0206: Long-Term RVQ Training (Epoch {epoch})', fontsize=14)

    epochs = range(1, len(history['train_total_loss']) + 1)

    # ===== Row 1: Losses =====
    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, history['val_total_loss'], 'r-', label='Val', alpha=0.7)
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # 2. Commitment Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_loss_commit'], 'b-', label='Train Commit', alpha=0.7)
    ax.plot(epochs, history['val_loss_commit'], 'r-', label='Val Commit', alpha=0.7)
    ax.set_title('Commitment Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # 3. Quant Loss (post-quant MSE)
    ax = axes[0, 2]
    ax.plot(epochs, history['train_loss_quant'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, history['val_loss_quant'], 'r-', label='Val', alpha=0.7)
    ax.set_title('L_quant (post-quant MSE)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # ===== Row 2: Collapse Metrics + Intermediate =====
    # 4. Used Codes + Joint Diversity (dual axis)
    ax = axes[1, 0]
    if history.get('layer0_used_codes'):
        ax.plot(epochs, history['layer0_used_codes'], 'darkorange', linewidth=2,
                label='Used Codes')
        ax.set_ylabel('Used Codes', color='darkorange')
    ax2 = ax.twinx()
    if history.get('joint_diversity'):
        ax2.plot(epochs, history['joint_diversity'], 'teal', linewidth=2,
                 linestyle='--', label='Joint Div')
        ax2.axhline(y=0.30, color='orange', linestyle=':', alpha=0.5)
        ax2.axhline(y=0.7, color='red', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Joint Diversity', color='teal')
    ax.set_title('Used Codes & Joint Diversity')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    ax.grid(True)

    # 5. Feature MSE
    ax = axes[1, 1]
    if history.get('feature_mse'):
        ax.plot(epochs, history['feature_mse'], 'brown', linewidth=2)
        ax.axhline(y=0.1, color='red', linestyle='--', label='P2/P3 threshold')
        ax.set_title('Feature MSE (z_q vs t_e)')
        ax.legend()
    ax.set_xlabel('Epoch')
    ax.grid(True)

    # 6. Intermediate Loss + Weight
    ax = axes[1, 2]
    ax.plot(epochs, history['train_loss_inter'], 'b-', label='Train Inter', alpha=0.7)
    ax.plot(epochs, history['val_loss_inter'], 'r-', label='Val Inter', alpha=0.7)
    ax2 = ax.twinx()
    if history.get('intermediate_weight'):
        ax2.plot(epochs, history['intermediate_weight'], 'purple', linewidth=2,
                 linestyle='--', label='Weight')
        ax2.set_ylabel('Weight', color='purple')
    ax.set_title('Intermediate Loss & Weight')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper left')
    ax.grid(True)

    # ===== Row 3: Training Dynamics =====
    # 7. Per-Layer Intermediate Loss
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

    # 8. Curriculum Phase
    ax = axes[2, 1]
    if history.get('curriculum_phase'):
        ax.plot(epochs, history['curriculum_phase'], 'orange', linewidth=2)
    ax.set_title('Curriculum Phase')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Max Noise Ratio')
    ax.set_ylim(0, 1.1)
    ax.grid(True)

    # 9. Learning Rate
    ax = axes[2, 2]
    if history.get('lr'):
        ax.plot(epochs, history['lr'], 'green', linewidth=2)
        ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch{epoch:03d}.png', dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    """主函式：解析參數、載入資料與模型、執行 300 epochs 訓練"""

    parser = argparse.ArgumentParser(description='exp_0206: Long-Term RVQ Training')

    # Training basics
    parser.add_argument('--exp_name', type=str, default='longterm', help='實驗名稱')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--num_epochs', type=int, default=300, help='訓練 epochs 數')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=2, help='梯度累積步數')
    parser.add_argument('--lr', type=float, default=1e-4, help='學習率')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小學習率')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度')
    parser.add_argument('--device', type=str, default='cuda:0', help='裝置')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=256, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=512, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.2,
                        help='LoRA dropout (備註: TeacherStudentRVQ 使用 parent 預設 0.2，此參數記錄用)')

    # RVQ (Phase 3-2 6c best config)
    parser.add_argument('--n_rvq_layers', type=int, default=4, help='RVQ 層數')
    parser.add_argument('--rvq_codebook_size', type=int, default=2048, help='每層 codebook 大小')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay')
    parser.add_argument('--ema_dead_code_threshold', type=int, default=2, help='Dead-code reset threshold')
    parser.add_argument('--ema_usage_penalty', type=float, default=0.1, help='Usage penalty (log cluster_size)')

    # Loss weights
    parser.add_argument('--lambda_quant', type=float, default=1.0, help='Post-quant alignment weight')
    parser.add_argument('--lambda_pre', type=float, default=0.0, help='Pre-quant alignment weight (disabled)')
    parser.add_argument('--beta_commit', type=float, default=1.0, help='Encoder commitment weight')
    parser.add_argument('--lambda_codebook', type=float, default=0.0, help='Codebook gradient loss (EMA mode=0)')

    # Intermediate supervision
    parser.add_argument('--intermediate_weight', type=float, default=0.5, help='中間層監督權重（基礎）')
    parser.add_argument('--intermediate_weight_min', type=float, default=0.25, help='中間層監督權重（最小）')
    parser.add_argument('--warmdown_epochs', type=int, default=50, help='Warmdown 持續 epochs')
    parser.add_argument('--intermediate_L3_weight', type=float, default=0.3, help='L3 層權重')
    parser.add_argument('--intermediate_L4_weight', type=float, default=0.5, help='L4 層權重')
    parser.add_argument('--intermediate_L6_weight', type=float, default=0.5, help='L6 層權重')

    # Curriculum
    parser.add_argument('--curriculum_start', type=float, default=0.3, help='Curriculum 起始比例')
    parser.add_argument('--curriculum_end', type=float, default=0.85, help='Curriculum 結束比例')
    parser.add_argument('--curriculum_epochs', type=int, default=200, help='Curriculum 持續 epochs')

    # Saving
    parser.add_argument('--save_checkpoint_every', type=int, default=10, help='每 N epochs 存 checkpoint')
    parser.add_argument('--save_audio_interval', type=int, default=50, help='每 N epochs 存音檔')
    parser.add_argument('--eval_max_batches', type=int, default=50, help='評估最大批次數')

    args = parser.parse_args()

    # ===== Setup =====
    set_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(f'families/compat_legacy/plan_ori_vq/runs/{args.exp_name}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(exp_dir)

    # Config dict
    config = vars(args)
    config['timestamp'] = timestamp
    config['rvq_update'] = 'ema'  # 固定使用 EMA
    config['intermediate_indices'] = [3, 4, 6]  # 與 exp_k_v6 一致: L3, L4, L6

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_0206: Long-Term RVQ Training")
    print("=" * 70)
    print(f"實驗名稱: {args.exp_name}_{timestamp}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LR: {args.lr} → {args.min_lr}")
    print(f"RVQ: {args.n_rvq_layers} layers × K={args.rvq_codebook_size} (EMA, th={args.ema_dead_code_threshold})")
    print(f"Usage penalty: {args.ema_usage_penalty}")
    print(f"Loss: λ_quant={args.lambda_quant}, λ_pre={args.lambda_pre}, "
          f"β_commit={args.beta_commit}, inter={args.intermediate_weight}→{args.intermediate_weight_min}")
    print(f"Curriculum: {args.curriculum_start}→{args.curriculum_end} over {args.curriculum_epochs} epochs")
    print(f"Output: {exp_dir}")
    if log_path:
        print(f"Log: {log_path}")
    print("=" * 70)

    # ===== CUDA =====
    device = torch.device(args.device)
    cuda_preinit(device)

    # ===== Data =====
    print("\n載入資料...")
    phase_increment = (args.curriculum_end - args.curriculum_start) / (args.curriculum_epochs / 10)

    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=4,
        compute_snr=False,  # 加速載入
        initial_phase=args.curriculum_start,
        phase_increment=phase_increment,
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # ===== Model =====
    print("\n建立模型...")
    model = TeacherStudentRVQ(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        intermediate_indices=[3, 4, 6],  # 與 exp_k_v6 一致
        device=device,
        n_rvq_layers=args.n_rvq_layers,
        rvq_codebook_size=args.rvq_codebook_size,
        rvq_update='ema',
        ema_decay=args.ema_decay,
        ema_dead_code_threshold=args.ema_dead_code_threshold,
        ema_usage_penalty=args.ema_usage_penalty,
    )

    inter_loss_fn = IntermediateSupervisionLossV6(
        layer_weights={
            3: args.intermediate_L3_weight,
            4: args.intermediate_L4_weight,
            6: args.intermediate_L6_weight,
        },
    )

    # ===== Optimizer & Scheduler =====
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Warmup + Cosine annealing (與 exp_k_v6 一致)
    def lr_lambda(epoch):
        """計算 learning rate 的縮放係數

        Args:
            epoch: 當前 epoch

        Returns:
            學習率縮放因子
        """
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)
        return max(args.min_lr / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler() if args.use_amp else None

    # ===== Training History =====
    history = {
        'train_total_loss': [], 'train_loss_quant': [], 'train_loss_pre': [],
        'train_loss_inter': [], 'train_loss_commit': [], 'train_loss_codebook': [],
        'train_intermediate_L3_loss': [], 'train_intermediate_L4_loss': [],
        'train_intermediate_L6_loss': [],
        'val_total_loss': [], 'val_loss_quant': [], 'val_loss_pre': [],
        'val_loss_inter': [], 'val_loss_commit': [], 'val_loss_codebook': [],
        'layer0_entropy': [], 'layer0_top10_mass': [], 'layer0_used_codes': [],
        'joint_diversity': [], 'feature_mse': [],
        'intermediate_weight': [], 'lr': [],
        'curriculum_phase': [],
        'p2_pass': [], 'p3_pass': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0

    # ===== Training Loop =====
    print(f"\n開始訓練 ({args.num_epochs} epochs)...")
    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        # Curriculum phase advance (每 10 epochs)
        if epoch > 1 and (epoch - 1) % 10 == 0 and curriculum_sampler is not None:
            curriculum_sampler.advance_phase()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, inter_loss_fn, device, epoch,
            config=config, scaler=scaler, curriculum_sampler=curriculum_sampler,
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Evaluate (每個 epoch 都做 collapse metrics)
        val_metrics = evaluate_epoch(
            model, val_loader, inter_loss_fn, device, config,
            max_batches=args.eval_max_batches,
        )

        epoch_time = time.time() - epoch_start

        # 記錄歷史
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_loss_quant'].append(train_metrics['loss_quant'])
        history['train_loss_pre'].append(train_metrics['loss_pre'])
        history['train_loss_inter'].append(train_metrics['loss_inter'])
        history['train_loss_commit'].append(train_metrics['loss_commit'])
        history['train_loss_codebook'].append(train_metrics['loss_codebook'])
        history['train_intermediate_L3_loss'].append(train_metrics.get('intermediate_L3_loss', 0))
        history['train_intermediate_L4_loss'].append(train_metrics.get('intermediate_L4_loss', 0))
        history['train_intermediate_L6_loss'].append(train_metrics.get('intermediate_L6_loss', 0))
        history['val_total_loss'].append(val_metrics['val_total_loss'])
        history['val_loss_quant'].append(val_metrics['val_loss_quant'])
        history['val_loss_pre'].append(val_metrics['val_loss_pre'])
        history['val_loss_inter'].append(val_metrics['val_loss_inter'])
        history['val_loss_commit'].append(val_metrics['val_loss_commit'])
        history['val_loss_codebook'].append(val_metrics['val_loss_codebook'])
        history['layer0_entropy'].append(val_metrics['layer0_entropy'])
        history['layer0_top10_mass'].append(val_metrics['layer0_top10_mass'])
        history['layer0_used_codes'].append(val_metrics['layer0_used_codes'])
        history['joint_diversity'].append(val_metrics['joint_diversity'])
        history['feature_mse'].append(val_metrics['feature_mse'])
        history['intermediate_weight'].append(train_metrics['intermediate_weight'])
        history['lr'].append(current_lr)
        history['curriculum_phase'].append(
            curriculum_sampler.current_phase if curriculum_sampler else args.curriculum_end
        )
        history['p2_pass'].append(val_metrics['p2_pass'])
        history['p3_pass'].append(val_metrics['p3_pass'])

        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_metrics['total_loss']:.4f} "
              f"quant={train_metrics['loss_quant']:.4f} "
              f"inter={train_metrics['loss_inter']:.4f} "
              f"commit={train_metrics['loss_commit']:.4f}")
        print(f"  Val:   loss={val_metrics['val_total_loss']:.4f} "
              f"quant={val_metrics['val_loss_quant']:.4f}")
        print(f"  Collapse: entropy={val_metrics['layer0_entropy']:.2f} "
              f"top10={val_metrics['layer0_top10_mass']:.4f} "
              f"used={val_metrics['layer0_used_codes']}/{args.rvq_codebook_size} "
              f"joint={val_metrics['joint_diversity']:.4f} "
              f"mse={val_metrics['feature_mse']:.4f}")
        print(f"  P2={val_metrics['p2_pass']} P3={val_metrics['p3_pass']} "
              f"LR={current_lr:.2e} IW={train_metrics['intermediate_weight']:.3f}")
        print(f"{'='*70}")

        # Best model
        if val_metrics['val_total_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_total_loss']
            best_epoch = epoch
            best_path = exp_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'rvq_state_dict': model.rvq.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'metrics': val_metrics,
                'config': config,
            }, best_path)
            print(f"  ✅ Best model saved (epoch {epoch}, val_loss={best_val_loss:.4f})")

        # Periodic checkpoint (LoRA + RVQ only, 節省空間)
        if epoch % args.save_checkpoint_every == 0:
            ckpt_dir = exp_dir / 'checkpoints'
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f'checkpoint_epoch{epoch:03d}.pt'
            torch.save({
                'epoch': epoch,
                **get_lora_rvq_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, ckpt_path)
            print(f"  📦 Checkpoint saved: {ckpt_path.name}")

        # Audio samples (含 epoch 1，與 baseline v6 一致)
        if epoch % args.save_audio_interval == 0 or epoch == 1:
            print("  🎵 儲存音檔樣本...")
            save_audio_samples(model, val_loader, device, exp_dir, epoch,
                               num_samples=2, split='val')
            save_audio_samples(model, train_loader, device, exp_dir, epoch,
                               num_samples=2, split='train')

        # Training curves
        if epoch % args.save_checkpoint_every == 0:
            plot_training_curves(history, exp_dir, epoch)

        # Save history (每個 epoch 更新)
        with open(exp_dir / 'metrics_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # ===== Final Summary =====
    print("\n" + "=" * 70)
    print("訓練完成!")
    print("=" * 70)

    final_metrics = {
        'epoch': args.num_epochs,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_layer0_entropy': history['layer0_entropy'][-1],
        'final_layer0_top10_mass': history['layer0_top10_mass'][-1],
        'final_layer0_used_codes': history['layer0_used_codes'][-1],
        'final_joint_diversity': history['joint_diversity'][-1],
        'final_feature_mse': history['feature_mse'][-1],
        'final_p2_pass': history['p2_pass'][-1],
        'final_p3_pass': history['p3_pass'][-1],
    }

    summary = {
        'config': config,
        'final_metrics': final_metrics,
        'acceptance': {
            'P2_pass': history['p2_pass'][-1],
            'P3_pass': history['p3_pass'][-1],
        },
        'baseline_reference': {
            'exp_k_v6_single_vq_entropy': 6.07,
            'exp_k_v6_single_vq_top10_mass': 0.197,
            'exp_k_v6_single_vq_used_codes': 740,
            'note': 'Baseline uses single VQ K=4096; not directly comparable to RVQ K=2048×4layers',
        },
        'phase3_2_reference': {
            '6c_up01_entropy': 9.03,
            '6c_up01_top10_mass': 0.158,
            '6c_up01_used_codes': 1089,
            '6c_up01_feature_mse': 0.034,
            'note': 'Phase 3-2 short-run (1000 steps); this exp extends to 300 epochs',
        },
    }

    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Final training curves
    plot_training_curves(history, exp_dir, args.num_epochs)

    # Save final model
    final_path = exp_dir / 'final_model.pt'
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'rvq_state_dict': model.rvq.state_dict(),
        'metrics': final_metrics,
        'config': config,
    }, final_path)

    print(f"\n最終模型: {final_path}")
    print(f"最佳模型: best_model.pt (epoch {best_epoch})")
    print(f"Summary: {exp_dir / 'summary.json'}")
    print(f"\n結果:")
    print(f"  Layer0 Entropy: {final_metrics['final_layer0_entropy']:.2f}")
    print(f"  Layer0 Top-10:  {final_metrics['final_layer0_top10_mass']:.4f}")
    print(f"  Used Codes:     {final_metrics['final_layer0_used_codes']}")
    print(f"  Joint Div:      {final_metrics['final_joint_diversity']:.4f}")
    print(f"  Feature MSE:    {final_metrics['final_feature_mse']:.4f}")
    print(f"  P2 Pass:        {final_metrics['final_p2_pass']}")
    print(f"  P3 Pass:        {final_metrics['final_p3_pass']}")
    print("=" * 70)


if __name__ == '__main__':
    main()

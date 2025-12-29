"""
Exp67: Curriculum Learning + VQ-Aware Loss 組合

結合兩個有效策略：
1. Exp64 Curriculum Learning (Best Val Acc: 1.06%) - 從簡單到困難
2. Exp63 VQ-Aware Loss (Best Val Acc: 0.95%) - 改善 VQ 量化品質

假設：
- Curriculum 讓模型先學會簡單的 denoising
- VQ-Aware Loss 確保 features 更接近正確的 codebook
- 兩者結合可能達到更好效果
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
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX, TRAIN_CACHE, VAL_CACHE
from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1219.losses import compute_masked_accuracy
from exp_1226.losses import MaskedCombinedLossV3  # V3 支援 VQ-Aware Loss
from exp_1226.data_curriculum import (
    create_curriculum_dataloaders,
    CurriculumDataset,
    collate_fn_curriculum
)


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_trainable_params(model):
    return (p for p in model.parameters() if p.requires_grad)


def verify_model_state(model, stage: str):
    if model.teacher.training:
        raise RuntimeError(f"[{stage}] Teacher 意外進入 train 模式!")
    if model.teacher.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Teacher quantizer 意外進入 train 模式!")
    if model.student.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Student quantizer 意外進入 train 模式!")


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                distance_matrix, encoder_stride=320, scaler=None, use_amp=True,
                check_interval=100, grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'vq_commitment_loss': 0, 'vq_distortion_loss': 0,  # VQ-Aware metrics
        'masked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
        'avg_snr': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                output = model(noisy_audio, clean_audio)
                loss, loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                )

            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(noisy_audio, clean_audio)
            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
            )

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        if (batch_idx + 1) % check_interval == 0:
            verify_model_state(model, f"Epoch {epoch} Batch {batch_idx + 1}")

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['vq_commitment_loss'] += loss_info.get('vq_commitment_loss', 0)
        metrics['vq_distortion_loss'] += loss_info.get('vq_distortion_loss', 0)

        # 追蹤 SNR
        if 'snr' in batch:
            metrics['avg_snr'] += batch['snr'].mean().item()

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, correct, total = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += total
        metrics['total_frames'] += s_codes.numel()

        with torch.no_grad():
            s_flat = s_codes.reshape(-1).long()
            t_flat = t_codes.reshape(-1).long()
            dist = distance_matrix[s_flat, t_flat].mean().item()
            metrics['distance_loss'] += dist

        n_batches += 1
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'm_acc': f"{masked_acc*100:.2f}%",
            'vq_c': f"{loss_info.get('vq_commitment_loss', 0):.3f}"
        })

    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, distance_matrix,
             encoder_stride=320, use_amp=True):
    model.eval()
    verify_model_state(model, "Validation")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'vq_commitment_loss': 0, 'vq_distortion_loss': 0,
        'masked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)
            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
            )

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['vq_commitment_loss'] += loss_info.get('vq_commitment_loss', 0)
        metrics['vq_distortion_loss'] += loss_info.get('vq_distortion_loss', 0)

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, correct, total = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += total
        metrics['total_frames'] += s_codes.numel()

        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        dist = distance_matrix[s_flat, t_flat].mean().item()
        metrics['distance_loss'] += dist

        n_batches += 1

    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=2, split='val'):
    model.eval()
    audio_dir = exp_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    data_iter = iter(dataloader)
    torch.cuda.empty_cache()

    for i in range(min(num_samples, len(dataloader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        noisy_audio = batch['noisy_audio'][:1].to(device)
        clean_audio = batch['clean_audio'][:1].to(device)

        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)

        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'), noisy_audio.cpu(), sample_rate)
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'), clean_audio.cpu(), sample_rate)

        try:
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_student_recon.wav'), student_recon.cpu(), sample_rate)
            del student_features, student_recon
            torch.cuda.empty_cache()

            teacher_features, _, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(device))
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_teacher_recon.wav'), teacher_recon.cpu(), sample_rate)
            del teacher_features, teacher_recon
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM when saving audio sample {i+1}, skipping")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples")


def plot_metrics(history, exp_dir):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Total Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    # Masked Accuracy
    ax = axes[0, 1]
    ax.plot([x * 100 for x in history['train_masked_acc']], label='Train')
    ax.plot([x * 100 for x in history['val_masked_acc']], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Masked Token Accuracy')
    ax.legend()
    ax.grid(True)

    # Feature Loss
    ax = axes[0, 2]
    ax.plot(history['train_feature_loss'], label='Train')
    ax.plot(history['val_feature_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature Loss')
    ax.legend()
    ax.grid(True)

    # Triplet Loss
    ax = axes[1, 0]
    ax.plot(history['train_triplet_loss'], label='Train')
    ax.plot(history['val_triplet_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Triplet Loss')
    ax.set_title('Triplet Loss')
    ax.legend()
    ax.grid(True)

    # VQ Commitment Loss (NEW)
    ax = axes[1, 1]
    ax.plot(history['train_vq_commitment_loss'], label='Train')
    ax.plot(history['val_vq_commitment_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('VQ Commitment Loss')
    ax.set_title('VQ Commitment Loss')
    ax.legend()
    ax.grid(True)

    # VQ Distortion Loss (NEW)
    ax = axes[1, 2]
    ax.plot(history['train_vq_distortion_loss'], label='Train')
    ax.plot(history['val_vq_distortion_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('VQ Distortion Loss')
    ax.set_title('VQ Distortion Loss')
    ax.legend()
    ax.grid(True)

    # Curriculum Phase
    ax = axes[2, 0]
    ax.plot(history['curriculum_phase'], color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Data Ratio')
    ax.set_title('Curriculum Phase (Data %)')
    ax.set_ylim(0, 1.1)
    ax.grid(True)

    # Average SNR
    ax = axes[2, 1]
    ax.plot(history['train_avg_snr'], label='Train Avg SNR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Training Data Average SNR')
    ax.legend()
    ax.grid(True)

    # Distance
    ax = axes[2, 2]
    ax.plot(history['train_dist'], label='Train')
    ax.plot(history['val_dist'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Average Distance')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Exp67: Curriculum + VQ-Aware')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp67_curriculum_vq')
    parser.add_argument('--output_dir', type=str, default=None)

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_layers', type=str, default='all_18')

    # Loss weights
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=0.0)

    # VQ-Aware Loss (from Exp63)
    parser.add_argument('--vq_commitment_weight', type=float, default=0.1)
    parser.add_argument('--vq_distortion_weight', type=float, default=0.1)
    parser.add_argument('--vq_temperature', type=float, default=1.0)

    # Curriculum Learning (from Exp64)
    parser.add_argument('--curriculum_mode', type=str, default='curriculum',
                        choices=['curriculum', 'anti_curriculum'])
    parser.add_argument('--initial_phase', type=float, default=0.3)
    parser.add_argument('--phase_increment', type=float, default=0.1)
    parser.add_argument('--phase_advance_epochs', type=int, default=30)

    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--check_interval', type=int, default=100)

    # Scheduler
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Gradient Accumulation
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=50)

    parser.add_argument('--encoder_stride', type=int, default=320)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(__file__).parent / 'runs' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"Exp67: Curriculum Learning + VQ-Aware Loss")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)
    print(f"Curriculum Config:")
    print(f"  Mode: {args.curriculum_mode}")
    print(f"  Initial Phase: {args.initial_phase:.0%}")
    print(f"  Phase Increment: {args.phase_increment:.0%}")
    print(f"  Advance Every: {args.phase_advance_epochs} epochs")
    print(f"VQ-Aware Config:")
    print(f"  VQ Commitment Weight: {args.vq_commitment_weight}")
    print(f"  VQ Distortion Weight: {args.vq_distortion_weight}")
    print(f"  VQ Temperature: {args.vq_temperature}")
    print("=" * 60)

    # Load data with curriculum
    print("\n載入資料 (with curriculum)...")

    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        curriculum_mode=args.curriculum_mode,
        initial_phase=args.initial_phase,
        phase_increment=args.phase_increment,
        compute_snr=True,
    )
    print(f"  Initial curriculum: {len(curriculum_sampler)} samples ({args.initial_phase:.0%})")
    print(f"  Val batches: {len(val_loader)}")

    # Load distance matrix
    print("\n載入距離矩陣...")
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)

    # Create model
    print("\n創建模型...")
    model = TeacherStudentConfigurableLoRA(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Loss Function - V3 支援 VQ-Aware
    loss_fn = MaskedCombinedLossV3(
        feature_weight=args.feature_weight,
        cosine_weight=args.cosine_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        ce_weight=args.ce_weight,
        vq_commitment_weight=args.vq_commitment_weight,
        vq_distortion_weight=args.vq_distortion_weight,
        vq_temperature=args.vq_temperature,
        encoder_stride=args.encoder_stride,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=args.lr * 0.01
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )

    scaler = GradScaler() if args.use_amp else None

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_masked_acc': [], 'val_masked_acc': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_triplet_loss': [], 'val_triplet_loss': [],
        'train_vq_commitment_loss': [], 'val_vq_commitment_loss': [],
        'train_vq_distortion_loss': [], 'val_vq_distortion_loss': [],
        'train_dist': [], 'val_dist': [],
        'train_avg_snr': [],
        'curriculum_phase': [],
        'lr': [],
    }

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    last_epoch = 0

    # Training loop
    print("\n開始訓練...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs} (Curriculum: {curriculum_sampler.current_phase:.0%})")
        print(f"{'='*60}")

        # Advance curriculum phase
        if epoch > 1 and (epoch - 1) % args.phase_advance_epochs == 0:
            curriculum_sampler.advance_phase()

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, args.encoder_stride, scaler, args.use_amp,
            args.check_interval, args.grad_clip, args.gradient_accumulation_steps
        )

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        if scheduler is not None:
            scheduler.step()

        val_metrics = validate(
            model, val_loader, loss_fn, device,
            distance_matrix, args.encoder_stride, args.use_amp
        )

        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_masked_acc'].append(train_metrics['masked_acc'])
        history['val_masked_acc'].append(val_metrics['masked_acc'])
        history['train_feature_loss'].append(train_metrics['feature_loss'])
        history['val_feature_loss'].append(val_metrics['feature_loss'])
        history['train_triplet_loss'].append(train_metrics['triplet_loss'])
        history['val_triplet_loss'].append(val_metrics['triplet_loss'])
        history['train_vq_commitment_loss'].append(train_metrics['vq_commitment_loss'])
        history['val_vq_commitment_loss'].append(val_metrics['vq_commitment_loss'])
        history['train_vq_distortion_loss'].append(train_metrics['vq_distortion_loss'])
        history['val_vq_distortion_loss'].append(val_metrics['vq_distortion_loss'])
        history['train_dist'].append(train_metrics['distance_loss'])
        history['val_dist'].append(val_metrics['distance_loss'])
        history['train_avg_snr'].append(train_metrics['avg_snr'])
        history['curriculum_phase'].append(curriculum_sampler.current_phase)

        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, "
              f"Masked Acc={train_metrics['masked_acc']*100:.2f}%, "
              f"Avg SNR={train_metrics['avg_snr']:.1f} dB")
        print(f"       VQ Commit={train_metrics['vq_commitment_loss']:.4f}, "
              f"VQ Distort={train_metrics['vq_distortion_loss']:.4f}")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, "
              f"Masked Acc={val_metrics['masked_acc']*100:.2f}%")
        print(f"       Distance={val_metrics['distance_loss']:.4f}")

        last_epoch = epoch  # Track last completed epoch

        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_masked_acc': val_metrics['masked_acc'],
                'curriculum_phase': curriculum_sampler.current_phase,
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best model saved! Masked Acc: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs (best: {best_val_acc*100:.2f}% @ epoch {best_epoch})")

        # Early stopping check
        if patience_counter >= args.early_stopping_patience:
            print(f"\n⚠ Early stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
            # Save last epoch model before stopping
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_masked_acc': val_metrics['masked_acc'],
                'curriculum_phase': curriculum_sampler.current_phase,
                'config': config,
                'early_stopped': True,
            }, exp_dir / 'last_model.pt')
            print(f"  ✓ Last epoch model saved to: {exp_dir / 'last_model.pt'}")
            break

        if epoch % 50 == 0 or epoch == 1:
            save_audio_samples(model, train_loader, device, exp_dir, epoch, num_samples=2, split='train')
            save_audio_samples(model, val_loader, device, exp_dir, epoch, num_samples=2, split='val')

        plot_metrics(history, exp_dir)

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # Save final model if training completed without early stopping
    if patience_counter < args.early_stopping_patience:
        torch.save({
            'epoch': last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_masked_acc': val_metrics['masked_acc'],
            'curriculum_phase': curriculum_sampler.current_phase,
            'config': config,
            'early_stopped': False,
        }, exp_dir / 'last_model.pt')
        print(f"\n✓ Final model saved to: {exp_dir / 'last_model.pt'}")

    print("\n" + "=" * 60)
    print("訓練完成!")
    print(f"Last Epoch: {last_epoch}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Masked Acc: {best_val_acc*100:.2f}%")
    print(f"Early Stopped: {patience_counter >= args.early_stopping_patience}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

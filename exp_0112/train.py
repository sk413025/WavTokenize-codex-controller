"""
exp_0112: 三區差異化訓練 (Exp I)

Exp I: 中層強化訓練 - 三區差異化 LR + 淺層輕度 L2

配置:
- 淺層 (L0-L4):  LR=5e-6 (0.5x), L2=0.01
- 中層 (L5-L8):  LR=2e-5 (2.0x), L2=0     ← 重點學習
- 深層 (L9-L17): LR=1e-5 (1.0x), L2=0

執行:
    python exp_0112/train.py --exp_name exp_i_mid_focus
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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX, TRAIN_CACHE, VAL_CACHE
from exp_0112.models import TeacherStudentThreeZone
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy
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
                check_interval=100, grad_clip=1.0, gradient_accumulation_steps=1,
                shallow_l2_weight=0.01):
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'masked_acc': 0, 'distance_loss': 0, 'shallow_l2_loss': 0,
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

                # 淺層 L2 正則化
                if shallow_l2_weight > 0:
                    shallow_l2_loss = model.compute_shallow_l2_regularization()
                    loss = loss + shallow_l2_weight * shallow_l2_loss
                    metrics['shallow_l2_loss'] += shallow_l2_loss.item()

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

            if shallow_l2_weight > 0:
                shallow_l2_loss = model.compute_shallow_l2_regularization()
                loss = loss + shallow_l2_weight * shallow_l2_loss
                metrics['shallow_l2_loss'] += shallow_l2_loss.item()

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
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'm_acc': f"{masked_acc*100:.2f}%"})

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
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM when saving audio sample {i+1}, skipping")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples")


def plot_metrics(history, exp_dir):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

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
    ax = axes[0, 3]
    ax.plot(history['train_triplet_loss'], label='Train')
    ax.plot(history['val_triplet_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Triplet Loss')
    ax.set_title('Triplet Loss')
    ax.legend()
    ax.grid(True)

    # Train-Val Gap
    ax = axes[1, 0]
    gap = [t - v for t, v in zip(history['train_masked_acc'], history['val_masked_acc'])]
    ax.plot([g * 100 for g in gap], color='red')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (%)')
    ax.set_title('Train-Val Accuracy Gap')
    ax.grid(True)

    # Distance
    ax = axes[1, 1]
    ax.plot(history['train_dist'], label='Train')
    ax.plot(history['val_dist'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Average Distance')
    ax.legend()
    ax.grid(True)

    # Learning Rates (Three zones)
    ax = axes[1, 2]
    if 'lr_shallow' in history and history['lr_shallow']:
        ax.plot(history['lr_shallow'], label='Shallow LR', linestyle='-')
        ax.plot(history['lr_middle'], label='Middle LR', linestyle='--')
        ax.plot(history['lr_deep'], label='Deep LR', linestyle=':')
        ax.legend()
    else:
        ax.plot(history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Three-Zone Learning Rates')
    ax.grid(True)

    # Shallow L2 Loss
    ax = axes[1, 3]
    if 'train_shallow_l2_loss' in history and any(x > 0 for x in history['train_shallow_l2_loss']):
        ax.plot(history['train_shallow_l2_loss'], label='Shallow L2 Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L2 Loss')
        ax.set_title('Shallow Layer L2 Regularization')
        ax.legend()
    else:
        ax.plot(history['curriculum_phase'], color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Data Ratio')
        ax.set_title('Curriculum Phase')
        ax.set_ylim(0, 1.1)
    ax.grid(True)

    plt.suptitle('Exp I: Three-Zone Mid-Focus Training', fontsize=14)
    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Exp 0112: 三區差異化訓練 (Exp I)')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp_i_mid_focus')
    parser.add_argument('--output_dir', type=str, default=None)

    # LoRA Config
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)

    # Three-Zone Learning Rates
    parser.add_argument('--lr_shallow', type=float, default=5e-6,
                        help='Learning rate for shallow layers (L0-L4)')
    parser.add_argument('--lr_middle', type=float, default=2e-5,
                        help='Learning rate for middle layers (L5-L8) - FOCUS ZONE')
    parser.add_argument('--lr_deep', type=float, default=1e-5,
                        help='Learning rate for deep layers (L9-L17)')

    # Shallow L2 Regularization
    parser.add_argument('--shallow_l2_weight', type=float, default=0.01,
                        help='L2 regularization weight for shallow layers')

    # Loss weights
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=0.0)

    # Curriculum Learning
    parser.add_argument('--curriculum_mode', type=str, default='curriculum',
                        choices=['curriculum', 'anti_curriculum'])
    parser.add_argument('--initial_phase', type=float, default=0.3)
    parser.add_argument('--phase_increment', type=float, default=0.1)
    parser.add_argument('--phase_advance_epochs', type=int, default=30)

    # Training
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
    config['experiment_type'] = 'Exp I: Three-Zone Mid-Focus Training'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"Exp 0112: {config['experiment_type']}")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)
    print(f"\nThree-Zone Configuration:")
    print(f"  Shallow (L0-L4): LR={args.lr_shallow:.1e}, L2={args.shallow_l2_weight}")
    print(f"  Middle (L5-L8):  LR={args.lr_middle:.1e}, L2=0 ★ Focus Zone")
    print(f"  Deep (L9-L17):   LR={args.lr_deep:.1e}, L2=0")
    print("=" * 60)

    # Load data
    print("\n載入資料 (with curriculum)...")
    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        curriculum_mode=args.curriculum_mode,
        initial_phase=args.initial_phase,
        phase_increment=args.phase_increment,
        compute_snr=False,
    )
    print(f"  Initial curriculum: {len(curriculum_sampler)} samples ({args.initial_phase:.0%})")
    print(f"  Val batches: {len(val_loader)}")

    # Load distance matrix
    print("\n載入距離矩陣...")
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)

    # Create model
    print("\n創建模型...")
    model = TeacherStudentThreeZone(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )

    # 三區差異化參數組
    param_groups = model.get_three_zone_param_groups(
        lr_shallow=args.lr_shallow,
        lr_middle=args.lr_middle,
        lr_deep=args.lr_deep,
        weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Loss Function
    loss_fn = MaskedCombinedLossV2(
        feature_weight=args.feature_weight,
        cosine_weight=args.cosine_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        ce_weight=args.ce_weight,
        encoder_stride=args.encoder_stride,
    )

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6
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
        'train_dist': [], 'val_dist': [],
        'train_avg_snr': [],
        'train_shallow_l2_loss': [],
        'curriculum_phase': [],
        'lr': [],
        'lr_shallow': [],
        'lr_middle': [],
        'lr_deep': [],
    }

    best_val_acc = 0
    best_epoch = 0

    # Training loop
    print("\n開始訓練...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs} (Curriculum: {curriculum_sampler.current_phase:.0%})")
        print(f"{'='*60}")

        if epoch > 1 and (epoch - 1) % args.phase_advance_epochs == 0:
            curriculum_sampler.advance_phase()

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, args.encoder_stride, scaler, args.use_amp,
            args.check_interval, args.grad_clip, args.gradient_accumulation_steps,
            shallow_l2_weight=args.shallow_l2_weight
        )

        # Record LRs
        history['lr_shallow'].append(optimizer.param_groups[0]['lr'])
        history['lr_middle'].append(optimizer.param_groups[1]['lr'])
        history['lr_deep'].append(optimizer.param_groups[2]['lr'])
        history['lr'].append(optimizer.param_groups[1]['lr'])  # Use middle LR as reference

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
        history['train_dist'].append(train_metrics['distance_loss'])
        history['val_dist'].append(val_metrics['distance_loss'])
        history['train_avg_snr'].append(train_metrics['avg_snr'])
        history['train_shallow_l2_loss'].append(train_metrics.get('shallow_l2_loss', 0))
        history['curriculum_phase'].append(curriculum_sampler.current_phase)

        train_val_gap = train_metrics['masked_acc'] - val_metrics['masked_acc']

        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, "
              f"Masked Acc={train_metrics['masked_acc']*100:.2f}%, "
              f"Avg SNR={train_metrics['avg_snr']:.1f} dB")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, "
              f"Masked Acc={val_metrics['masked_acc']*100:.2f}%")
        print(f"       Distance={val_metrics['distance_loss']:.4f}")
        print(f"       Train-Val Gap={train_val_gap*100:.2f}%")
        print(f"       Shallow L2 Loss={train_metrics['shallow_l2_loss']:.6f}")
        print(f"       LRs: shallow={optimizer.param_groups[0]['lr']:.2e}, "
              f"middle={optimizer.param_groups[1]['lr']:.2e}, "
              f"deep={optimizer.param_groups[2]['lr']:.2e}")

        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_masked_acc': val_metrics['masked_acc'],
                'curriculum_phase': curriculum_sampler.current_phase,
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best model saved! Masked Acc: {best_val_acc*100:.2f}%")

        if epoch % 50 == 0 or epoch == 1:
            save_audio_samples(model, train_loader, device, exp_dir, epoch, num_samples=2, split='train')
            save_audio_samples(model, val_loader, device, exp_dir, epoch, num_samples=2, split='val')

        plot_metrics(history, exp_dir)

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("訓練完成!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Masked Acc: {best_val_acc*100:.2f}%")
    print(f"Results saved to: {exp_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Exp66: Post-VQ Feature Loss 實驗

核心概念：
- 診斷發現：Pre-VQ Cosine Sim = 0.495，Post-VQ Cosine Sim = 0.9325
- 目前訓練只優化 Pre-VQ encoder output
- 直接優化 Post-VQ quantized features 應該能更有效改善音質

新增 Loss:
- Post-VQ Feature Loss: MSE(VQ(student), VQ(teacher))
- Post-VQ Cosine Loss: Cosine similarity between Post-VQ features
- 使用 Straight-Through Estimator 讓梯度穿過 VQ

配置 (基於 Exp55):
- Feature + Triplet Loss (基礎)
- 新增: Post-VQ Feature Loss (λ=0.5)
- 新增: Post-VQ Cosine Loss (λ=0.5)
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
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1212.data_aligned import create_aligned_dataloaders
from families.deps.wavtokenizer_core.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
from exp_1217.models import TeacherStudentConfigurableLoRA
from families.compat_legacy.curriculum_data.losses_post_vq import MaskedCombinedLossV5
from exp_1219.losses import compute_masked_accuracy


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


def forward_with_quantized(model, noisy_audio, clean_audio):
    """
    Extended forward that returns quantized features

    Returns dict with:
    - student_encoder_out: Pre-VQ encoder output
    - teacher_encoder_out: Pre-VQ encoder output
    - student_quantized: Post-VQ quantized features
    - teacher_quantized: Post-VQ quantized features
    - student_codes, teacher_codes, codebook
    """
    if noisy_audio.dim() == 2:
        noisy_audio = noisy_audio.unsqueeze(1)
    if clean_audio.dim() == 2:
        clean_audio = clean_audio.unsqueeze(1)

    # Teacher forward
    model.teacher.eval()
    model.teacher.feature_extractor.encodec.quantizer.eval()

    with torch.no_grad():
        teacher_encoder_out = model.teacher.feature_extractor.encodec.encoder(clean_audio)
        teacher_vq = model.teacher.feature_extractor.encodec.quantizer(
            teacher_encoder_out, frame_rate=75, bandwidth=0.075
        )
        teacher_codes = teacher_vq.codes
        teacher_quantized = teacher_vq.quantized  # Post-VQ features

    # Student forward
    student_encoder_out = model.student.feature_extractor.encodec.encoder(noisy_audio)

    model.student.feature_extractor.encodec.quantizer.eval()

    with torch.no_grad():
        quantizer = model.student.feature_extractor.encodec.quantizer
        student_vq = quantizer(student_encoder_out, frame_rate=75, bandwidth=0.075)
        student_codes = student_vq.codes
        student_quantized = student_vq.quantized  # Post-VQ features

    return {
        'student_encoder_out': student_encoder_out,
        'teacher_encoder_out': teacher_encoder_out,
        'student_quantized': student_quantized,
        'teacher_quantized': teacher_quantized,
        'student_codes': student_codes,
        'teacher_codes': teacher_codes,
        'codebook': model.codebook,
    }


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                distance_matrix, encoder_stride=320, scaler=None, use_amp=True,
                check_interval=100, grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'post_vq_feature_loss': 0, 'post_vq_cosine_loss': 0,
        'post_vq_cos_sim': 0,
        'masked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
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
                # Use extended forward with quantized features
                output = forward_with_quantized(model, noisy_audio, clean_audio)
                loss, loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                    student_quantized=output['student_quantized'],
                    teacher_quantized=output['teacher_quantized'],
                )

            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = forward_with_quantized(model, noisy_audio, clean_audio)
            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
                student_quantized=output['student_quantized'],
                teacher_quantized=output['teacher_quantized'],
            )

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        # Accumulate metrics
        metrics['total_loss'] += loss.item()
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['post_vq_feature_loss'] += loss_info.get('post_vq_feature_loss', 0)
        metrics['post_vq_cosine_loss'] += loss_info.get('post_vq_cosine_loss', 0)
        metrics['post_vq_cos_sim'] += loss_info.get('post_vq_cos_sim', 0)

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, correct, total = compute_masked_accuracy(
            s_codes, t_codes, lengths, encoder_stride
        )
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += total
        metrics['total_frames'] += s_codes.numel()

        with torch.no_grad():
            s_flat = s_codes.reshape(-1).long()
            t_flat = t_codes.reshape(-1).long()
            dist = distance_matrix[s_flat, t_flat].mean().item()
            metrics['distance_loss'] += dist

        n_batches += 1

        post_vq_sim = loss_info.get('post_vq_cos_sim', 0)
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'm_acc': f"{masked_acc*100:.1f}%",
            'pv_cos': f"{post_vq_sim:.2f}"
        })

        if (batch_idx + 1) % check_interval == 0:
            verify_model_state(model, f"Epoch {epoch} Batch {batch_idx}")

    # Average metrics
    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    return metrics


def validate_epoch(model, dataloader, loss_fn, device, epoch,
                   distance_matrix, encoder_stride=320, use_amp=True):
    model.eval()
    verify_model_state(model, f"Epoch {epoch} Val")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'post_vq_feature_loss': 0, 'post_vq_cosine_loss': 0,
        'post_vq_cos_sim': 0,
        'masked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
    }
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} Val"):
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)
            lengths = batch['lengths'].to(device)

            if use_amp:
                with autocast(enabled=use_amp):
                    output = forward_with_quantized(model, noisy_audio, clean_audio)
                    loss, loss_info = loss_fn(
                        student_features=output['student_encoder_out'],
                        teacher_features=output['teacher_encoder_out'],
                        teacher_codes=output['teacher_codes'],
                        codebook=output['codebook'],
                        lengths=lengths,
                        student_quantized=output['student_quantized'],
                        teacher_quantized=output['teacher_quantized'],
                    )
            else:
                output = forward_with_quantized(model, noisy_audio, clean_audio)
                loss, loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                    student_quantized=output['student_quantized'],
                    teacher_quantized=output['teacher_quantized'],
                )

            metrics['total_loss'] += loss.item()
            metrics['feature_loss'] += loss_info.get('feature_loss', 0)
            metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
            metrics['post_vq_feature_loss'] += loss_info.get('post_vq_feature_loss', 0)
            metrics['post_vq_cosine_loss'] += loss_info.get('post_vq_cosine_loss', 0)
            metrics['post_vq_cos_sim'] += loss_info.get('post_vq_cos_sim', 0)

            s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
            t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

            masked_acc, correct, total = compute_masked_accuracy(
                s_codes, t_codes, lengths, encoder_stride
            )
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


def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=3, split='val'):
    """保存音檔樣本 (參考 exp_1223 格式)"""
    model.eval()
    audio_dir = exp_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    import torchaudio
    sample_rate = 24000
    data_iter = iter(dataloader)

    torch.cuda.empty_cache()

    for i in range(min(num_samples, len(dataloader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        noisy = batch['noisy_audio'][:1].to(device)
        clean = batch['clean_audio'][:1].to(device)

        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)

        # Save noisy input
        torchaudio.save(
            str(audio_dir / f"sample_{i+1}_noisy.wav"),
            noisy.squeeze(1).cpu(), sample_rate
        )

        # Save clean target
        torchaudio.save(
            str(audio_dir / f"sample_{i+1}_clean.wav"),
            clean.squeeze(1).cpu(), sample_rate
        )

        try:
            with torch.no_grad():
                # Get Post-VQ features for reconstruction
                student_feat, student_codes, _ = model.student.feature_extractor(
                    noisy.squeeze(1), bandwidth_id=0
                )
                teacher_feat, teacher_codes, _ = model.teacher.feature_extractor(
                    clean.squeeze(1), bandwidth_id=0
                )

                # Decode
                student_recon = model.teacher.decode(
                    student_feat, bandwidth_id=torch.tensor([0]).to(device)
                )
                teacher_recon = model.teacher.decode(
                    teacher_feat, bandwidth_id=torch.tensor([0]).to(device)
                )

                torchaudio.save(
                    str(audio_dir / f"sample_{i+1}_student_recon.wav"),
                    student_recon.cpu(), sample_rate
                )
                torchaudio.save(
                    str(audio_dir / f"sample_{i+1}_teacher_recon.wav"),
                    teacher_recon.cpu(), sample_rate
                )

                del student_feat, teacher_feat, student_recon, teacher_recon
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Warning: Failed to save {split} sample {i+1}: {e}")

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples")


def plot_metrics(history, exp_dir):
    """即時繪製訓練曲線"""
    if not history['train']:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs = range(1, len(history['train']) + 1)

    # Extract metrics
    train_loss = [h['total_loss'] for h in history['train']]
    val_loss = [h['total_loss'] for h in history['val']]
    train_acc = [h['masked_acc'] * 100 for h in history['train']]
    val_acc = [h['masked_acc'] * 100 for h in history['val']]
    train_feature_loss = [h['feature_loss'] for h in history['train']]
    val_feature_loss = [h['feature_loss'] for h in history['val']]
    train_triplet_loss = [h['triplet_loss'] for h in history['train']]
    val_triplet_loss = [h['triplet_loss'] for h in history['val']]
    train_post_vq_cos = [h['post_vq_cos_sim'] for h in history['train']]
    val_post_vq_cos = [h['post_vq_cos_sim'] for h in history['val']]
    train_post_vq_feature = [h['post_vq_feature_loss'] for h in history['train']]
    val_post_vq_feature = [h['post_vq_feature_loss'] for h in history['val']]

    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label='Train')
    ax.plot(epochs, val_loss, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    # Masked Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, label='Train')
    ax.plot(epochs, val_acc, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Masked Token Accuracy')
    ax.legend()
    ax.grid(True)

    # Feature Loss
    ax = axes[0, 2]
    ax.plot(epochs, train_feature_loss, label='Train')
    ax.plot(epochs, val_feature_loss, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature Loss')
    ax.legend()
    ax.grid(True)

    # Triplet Loss
    ax = axes[1, 0]
    ax.plot(epochs, train_triplet_loss, label='Train')
    ax.plot(epochs, val_triplet_loss, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Triplet Loss')
    ax.set_title('Triplet Loss')
    ax.legend()
    ax.grid(True)

    # Post-VQ Cosine Similarity
    ax = axes[1, 1]
    ax.plot(epochs, train_post_vq_cos, label='Train')
    ax.plot(epochs, val_post_vq_cos, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Post-VQ Cosine Similarity')
    ax.legend()
    ax.grid(True)

    # Post-VQ Feature Loss
    ax = axes[1, 2]
    ax.plot(epochs, train_post_vq_feature, label='Train')
    ax.plot(epochs, val_post_vq_feature, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Post-VQ Feature Loss')
    ax.set_title('Post-VQ Feature Loss')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp66_post_vq')
    parser.add_argument('--output_dir', type=str, required=True)

    # Model
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_layers', type=str, default='all_18')

    # Loss weights - Pre-VQ (基礎)
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=0.0)
    parser.add_argument('--encoder_stride', type=int, default=320)

    # Loss weights - VQ-Aware (Exp63)
    parser.add_argument('--vq_commitment_weight', type=float, default=0.0)
    parser.add_argument('--vq_distortion_weight', type=float, default=0.0)
    parser.add_argument('--vq_temperature', type=float, default=1.0)

    # Loss weights - Post-VQ (新增)
    parser.add_argument('--post_vq_feature_weight', type=float, default=0.5)
    parser.add_argument('--post_vq_cosine_weight', type=float, default=0.5)

    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--early_stopping_patience', type=int, default=50)

    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"Exp66: Post-VQ Feature Loss")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)
    print(f"Loss Config:")
    print(f"  Pre-VQ Feature: {args.feature_weight}")
    print(f"  Triplet: {args.triplet_weight} (margin={args.triplet_margin})")
    print(f"  Post-VQ Feature: {args.post_vq_feature_weight} (NEW)")
    print(f"  Post-VQ Cosine: {args.post_vq_cosine_weight} (NEW)")
    print("=" * 60)

    # Load data
    print("\n載入資料...")

    class DataConfig:
        batch_size = args.batch_size
        num_workers = args.num_workers

    train_loader, val_loader = create_aligned_dataloaders(DataConfig())
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Load model
    print("\n載入模型...")
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

    # Loss Function - V5 with Post-VQ losses
    loss_fn = MaskedCombinedLossV5(
        feature_weight=args.feature_weight,
        cosine_weight=args.cosine_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        ce_weight=args.ce_weight,
        vq_commitment_weight=args.vq_commitment_weight,
        vq_distortion_weight=args.vq_distortion_weight,
        vq_temperature=args.vq_temperature,
        post_vq_feature_weight=args.post_vq_feature_weight,
        post_vq_cosine_weight=args.post_vq_cosine_weight,
        encoder_stride=args.encoder_stride,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # AMP
    scaler = GradScaler() if args.use_amp else None

    # Load distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=False).to(device)

    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}

    print("\n" + "=" * 60)
    print("開始訓練...")
    print("=" * 60)

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, args.encoder_stride, scaler, args.use_amp,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, epoch,
            distance_matrix, args.encoder_stride, args.use_amp
        )

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Log
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
              f"Acc: {train_metrics['masked_acc']*100:.2f}%, "
              f"Post-VQ Cos: {train_metrics['post_vq_cos_sim']:.3f}")
        print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
              f"Acc: {val_metrics['masked_acc']*100:.2f}%, "
              f"Post-VQ Cos: {val_metrics['post_vq_cos_sim']:.3f}")

        # Save best model
        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best model saved! Val Acc: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\n⚠ Early stopping at epoch {epoch}")
            # Save last epoch model before stopping
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['masked_acc'],
                'early_stopped': True,
            }, exp_dir / 'last_model.pt')
            print(f"  ✓ Last epoch model saved to: {exp_dir / 'last_model.pt'}")
            break

        # Periodic saves (every 50 epochs or epoch 1)
        if epoch % 50 == 0 or epoch == 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, exp_dir / f'checkpoint_epoch_{epoch}.pt')
            # Save train and val audio samples
            save_audio_samples(model, train_loader, device, exp_dir, epoch, num_samples=2, split='train')
            save_audio_samples(model, val_loader, device, exp_dir, epoch, num_samples=2, split='val')

        # Plot metrics (即時更新)
        plot_metrics(history, exp_dir)

        # Save history (每個 epoch 保存)
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # Save final model if training completed without early stopping
    if patience_counter < args.early_stopping_patience:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_metrics['masked_acc'],
            'early_stopped': False,
        }, exp_dir / 'last_model.pt')
        print(f"\n✓ Final model saved to: {exp_dir / 'last_model.pt'}")

    print("\n" + "=" * 60)
    print("訓練完成!")
    print(f"Last Epoch: {epoch}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Masked Acc: {best_val_acc*100:.2f}%")
    print(f"Early Stopped: {patience_counter >= args.early_stopping_patience}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

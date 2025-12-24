"""
exp_1222: Audio Domain Loss 訓練腳本

核心策略:
- 訓練時: Encoder features → Decode (bypass VQ) → Audio Loss
- 推論時: 可選擇 bypass VQ 或經過 VQ

Loss 組合:
- Audio-level: Multi-Resolution STFT + Mel Loss (主要)
- Feature-level: MSE + Triplet (可選輔助)
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

from exp_1212.data_aligned import create_aligned_dataloaders
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
from exp_1222.models import TeacherStudentAudioLoss
from exp_1222.losses import AudioDomainLoss


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


def compute_token_accuracy(student_codes, teacher_codes):
    """計算 token accuracy (監控用，不用於 loss)"""
    s = student_codes[0] if student_codes.dim() == 3 else student_codes
    t = teacher_codes[0] if teacher_codes.dim() == 3 else teacher_codes
    return (s == t).float().mean().item()


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    loss_fn: AudioDomainLoss,
    device: str,
    epoch: int,
    distance_matrix: torch.Tensor,
    scaler=None,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """訓練一個 epoch"""
    model.train()

    metrics = {
        'total_loss': 0, 'stft_loss': 0, 'mel_loss': 0,
        'feature_loss': 0, 'triplet_loss': 0,
        'token_acc': 0, 'distance': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                output = model(noisy_audio, clean_audio, return_audio=True)

                loss, loss_info = loss_fn(
                    pred_audio=output['denoised_audio'],
                    target_audio=clean_audio.squeeze(1) if clean_audio.dim() == 3 else clean_audio,
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                )

            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(noisy_audio, clean_audio, return_audio=True)

            loss, loss_info = loss_fn(
                pred_audio=output['denoised_audio'],
                target_audio=clean_audio.squeeze(1) if clean_audio.dim() == 3 else clean_audio,
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
            )

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        # 記錄指標
        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['stft_loss'] += loss_info.get('stft_loss', 0)
        metrics['mel_loss'] += loss_info.get('mel_loss', 0)
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)

        # Token accuracy (監控用)
        token_acc = compute_token_accuracy(output['student_codes'], output['teacher_codes'])
        metrics['token_acc'] += token_acc

        # Distance (監控用)
        with torch.no_grad():
            s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
            t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
            s_flat = s_codes.reshape(-1).long()
            t_flat = t_codes.reshape(-1).long()
            dist = distance_matrix[s_flat, t_flat].mean().item()
            metrics['distance'] += dist

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'stft': f"{loss_info.get('stft_loss', 0):.3f}",
            'mel': f"{loss_info.get('mel_loss', 0):.3f}",
            'acc': f"{token_acc*100:.1f}%"
        })

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    loss_fn: AudioDomainLoss,
    device: str,
    distance_matrix: torch.Tensor,
    use_amp: bool = True,
) -> dict:
    """驗證模型"""
    model.eval()

    metrics = {
        'total_loss': 0, 'stft_loss': 0, 'mel_loss': 0,
        'feature_loss': 0, 'triplet_loss': 0,
        'token_acc': 0, 'distance': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio, return_audio=True)

            loss, loss_info = loss_fn(
                pred_audio=output['denoised_audio'],
                target_audio=clean_audio.squeeze(1) if clean_audio.dim() == 3 else clean_audio,
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
            )

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['stft_loss'] += loss_info.get('stft_loss', 0)
        metrics['mel_loss'] += loss_info.get('mel_loss', 0)
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)

        token_acc = compute_token_accuracy(output['student_codes'], output['teacher_codes'])
        metrics['token_acc'] += token_acc

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        dist = distance_matrix[s_flat, t_flat].mean().item()
        metrics['distance'] += dist

        n_batches += 1

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=3, split='val'):
    """保存音頻樣本"""
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

        # 保存 noisy
        noisy_save = noisy_audio.squeeze() if noisy_audio.dim() > 2 else noisy_audio[0]
        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'), noisy_save.unsqueeze(0).cpu(), sample_rate)

        # 保存 clean
        clean_save = clean_audio.squeeze() if clean_audio.dim() > 2 else clean_audio[0]
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'), clean_save.unsqueeze(0).cpu(), sample_rate)

        try:
            # Denoised (bypass VQ)
            output = model(noisy_audio, clean_audio, return_audio=True)
            denoised = output['denoised_audio'][0]
            torchaudio.save(str(audio_dir / f'sample_{i+1}_denoised_bypass.wav'), denoised.unsqueeze(0).cpu(), sample_rate)

            # Denoised (with VQ) - for comparison
            denoised_vq = model.inference(noisy_audio, use_vq=True)[0]
            torchaudio.save(str(audio_dir / f'sample_{i+1}_denoised_vq.wav'), denoised_vq.unsqueeze(0).cpu(), sample_rate)

            del output, denoised, denoised_vq
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM when saving audio sample {i+1}, skipping")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples to {audio_dir}")


def plot_metrics(history, exp_dir):
    """繪製訓練曲線"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Total Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    # STFT Loss
    ax = axes[0, 1]
    ax.plot(history['train_stft_loss'], label='Train')
    ax.plot(history['val_stft_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('STFT Loss')
    ax.set_title('STFT Loss')
    ax.legend()
    ax.grid(True)

    # Mel Loss
    ax = axes[0, 2]
    ax.plot(history['train_mel_loss'], label='Train')
    ax.plot(history['val_mel_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mel Loss')
    ax.set_title('Mel Loss')
    ax.legend()
    ax.grid(True)

    # Token Accuracy (監控)
    ax = axes[1, 0]
    ax.plot([x * 100 for x in history['train_token_acc']], label='Train')
    ax.plot([x * 100 for x in history['val_token_acc']], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Token Accuracy (Monitor)')
    ax.legend()
    ax.grid(True)

    # Distance (監控)
    ax = axes[1, 1]
    ax.plot(history['train_dist'], label='Train')
    ax.plot(history['val_dist'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Token Distance (Monitor)')
    ax.legend()
    ax.grid(True)

    # Learning Rate
    ax = axes[1, 2]
    ax.plot(history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_title('Learning Rate')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    # 實驗名稱
    parser.add_argument('--exp_name', type=str, required=True)

    # LoRA 參數
    parser.add_argument('--lora_rank', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=256)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_layers', type=str, default='all_18',
                        choices=['all_18', 'critical_8', 'critical_10'])

    # Audio Loss 權重
    parser.add_argument('--stft_weight', type=float, default=1.0)
    parser.add_argument('--mel_weight', type=float, default=1.0)

    # Feature Loss 權重 (可選)
    parser.add_argument('--feature_weight', type=float, default=0.0,
                        help='Feature-level MSE loss weight (optional)')
    parser.add_argument('--triplet_weight', type=float, default=0.0,
                        help='Triplet margin loss weight (optional)')
    parser.add_argument('--triplet_margin', type=float, default=0.2)

    # 訓練參數
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    # Scheduler
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Resume
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    exp_dir = Path(__file__).parent / 'runs' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"exp_1222: Audio Domain Loss Training")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)
    print(f"LoRA Config:")
    print(f"  Rank: {args.lora_rank}, Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}, Layers: {args.lora_layers}")
    print(f"Loss Config:")
    print(f"  STFT: {args.stft_weight}, Mel: {args.mel_weight}")
    print(f"  Feature: {args.feature_weight}, Triplet: {args.triplet_weight}")
    print(f"Training Config:")
    print(f"  Batch Size: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  LR: {args.lr}, Epochs: {args.num_epochs}")
    print("=" * 60)

    # 載入資料
    print("\n載入資料...")

    class DataConfig:
        batch_size = args.batch_size
        num_workers = args.num_workers
        pin_memory = True

    train_loader, val_loader = create_aligned_dataloaders(DataConfig())
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 載入距離矩陣 (監控用)
    print("\n載入距離矩陣...")
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)

    # 創建模型
    print("\n創建模型...")
    model = TeacherStudentAudioLoss(
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
    print(f"  Total: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Loss Function
    loss_fn = AudioDomainLoss(
        stft_weight=args.stft_weight,
        mel_weight=args.mel_weight,
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Resume
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = Path(__file__).parent / resume_path

        if resume_path.exists():
            print(f"\n載入 checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.resume_epoch:
                start_epoch = args.resume_epoch
            elif 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            print(f"  從 Epoch {start_epoch} 繼續訓練")

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=args.lr * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
        print(f"\nScheduler: Cosine with {args.warmup_epochs} epoch warmup")

    # AMP
    scaler = GradScaler() if args.use_amp else None

    # History
    history = {
        'train_loss': [], 'val_loss': [],
        'train_stft_loss': [], 'val_stft_loss': [],
        'train_mel_loss': [], 'val_mel_loss': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_triplet_loss': [], 'val_triplet_loss': [],
        'train_token_acc': [], 'val_token_acc': [],
        'train_dist': [], 'val_dist': [],
        'lr': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0

    # Training loop
    print("\n開始訓練...")
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, scaler, args.use_amp, args.grad_clip, args.gradient_accumulation_steps
        )

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        if scheduler:
            scheduler.step()

        val_metrics = validate(model, val_loader, loss_fn, device, distance_matrix, args.use_amp)

        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_stft_loss'].append(train_metrics['stft_loss'])
        history['val_stft_loss'].append(val_metrics['stft_loss'])
        history['train_mel_loss'].append(train_metrics['mel_loss'])
        history['val_mel_loss'].append(val_metrics['mel_loss'])
        history['train_feature_loss'].append(train_metrics['feature_loss'])
        history['val_feature_loss'].append(val_metrics['feature_loss'])
        history['train_triplet_loss'].append(train_metrics['triplet_loss'])
        history['val_triplet_loss'].append(val_metrics['triplet_loss'])
        history['train_token_acc'].append(train_metrics['token_acc'])
        history['val_token_acc'].append(val_metrics['token_acc'])
        history['train_dist'].append(train_metrics['distance'])
        history['val_dist'].append(val_metrics['distance'])

        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f} (STFT={train_metrics['stft_loss']:.3f}, Mel={train_metrics['mel_loss']:.3f})")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f} (STFT={val_metrics['stft_loss']:.3f}, Mel={val_metrics['mel_loss']:.3f})")
        print(f"       Token Acc={val_metrics['token_acc']*100:.2f}%, Distance={val_metrics['distance']:.4f}")

        # Save best model (based on val loss)
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best model! Val Loss: {best_val_loss:.4f}")

        # Save audio samples
        if epoch % 10 == 0 or epoch == 1:
            save_audio_samples(model, val_loader, device, exp_dir, epoch, num_samples=3, split='val')

        # Plot & save history
        plot_metrics(history, exp_dir)
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("訓練完成!")
    print(f"Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

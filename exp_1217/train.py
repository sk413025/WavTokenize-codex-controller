"""
exp_1217: 可配置 LoRA 訓練腳本

支持:
1. 可配置 Loss 組合 (Feature, Triplet, CE)
2. 可配置 LoRA 層 (all_18 vs critical_8)
3. 可配置 LoRA rank
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

from exp_1212.data_aligned import create_aligned_dataloaders
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
from exp_1217.models import TeacherStudentConfigurableLoRA, CodebookDriftError
from exp_1212.losses_masked import (
    MaskedCombinedLoss, compute_masked_accuracy, create_length_mask
)


def set_seed(seed: int = 42):
    """固定隨機種子"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_trainable_params(model):
    """獲取可訓練參數"""
    return (p for p in model.parameters() if p.requires_grad)


def verify_model_state(model, stage: str):
    """驗證模型狀態是否正確"""
    if model.teacher.training:
        raise RuntimeError(f"[{stage}] Teacher 意外進入 train 模式!")

    if model.teacher.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Teacher quantizer 意外進入 train 模式!")

    if model.student.feature_extractor.encodec.quantizer.training:
        raise RuntimeError(f"[{stage}] Student quantizer 意外進入 train 模式!")


def compute_unmasked_accuracy(student_codes, teacher_codes):
    """計算未 mask 的 token accuracy"""
    s = student_codes[0] if student_codes.dim() == 3 else student_codes
    t = teacher_codes[0] if teacher_codes.dim() == 3 else teacher_codes
    return (s == t).float().mean().item()


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    loss_fn: MaskedCombinedLoss,
    device: str,
    epoch: int,
    distance_matrix: torch.Tensor,
    encoder_stride: int = 320,
    scaler=None,
    use_amp: bool = True,
    check_interval: int = 100,
    grad_clip: float = 1.0,
) -> dict:
    """訓練一個 epoch"""
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0, 'ce_loss': 0,
        'masked_acc': 0, 'unmasked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                output = model(noisy_audio, clean_audio)

                logits = None
                if loss_fn.ce_weight > 0:
                    logits = model.compute_ce_logits(output['student_encoder_out'])

                loss, loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                    logits=logits,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(noisy_audio, clean_audio)

            logits = None
            if loss_fn.ce_weight > 0:
                logits = model.compute_ce_logits(output['student_encoder_out'])

            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
                logits=logits,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
            optimizer.step()

        if (batch_idx + 1) % check_interval == 0:
            verify_model_state(model, f"Epoch {epoch} Batch {batch_idx + 1}")

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['ce_loss'] += loss_info.get('ce_loss', 0)

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, correct, total = compute_masked_accuracy(
            s_codes, t_codes, lengths, encoder_stride
        )
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += total
        metrics['total_frames'] += s_codes.numel()

        unmasked_acc = compute_unmasked_accuracy(output['student_codes'], output['teacher_codes'])
        metrics['unmasked_acc'] += unmasked_acc

        with torch.no_grad():
            s_flat = s_codes.reshape(-1).long()
            t_flat = t_codes.reshape(-1).long()
            dist = distance_matrix[s_flat, t_flat].mean().item()
            metrics['distance_loss'] += dist

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'm_acc': f"{masked_acc*100:.1f}%",
            'u_acc': f"{unmasked_acc*100:.1f}%"
        })

    for key in ['total_loss', 'feature_loss', 'triplet_loss', 'ce_loss',
                'masked_acc', 'unmasked_acc', 'distance_loss']:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    loss_fn: MaskedCombinedLoss,
    device: str,
    distance_matrix: torch.Tensor,
    encoder_stride: int = 320,
    use_amp: bool = True,
) -> dict:
    """驗證模型"""
    model.eval()
    verify_model_state(model, "Validation")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0, 'ce_loss': 0,
        'masked_acc': 0, 'unmasked_acc': 0, 'distance_loss': 0,
        'valid_frames': 0, 'total_frames': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)

            logits = None
            if loss_fn.ce_weight > 0:
                logits = model.compute_ce_logits(output['student_encoder_out'])

            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
                logits=logits,
            )

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['ce_loss'] += loss_info.get('ce_loss', 0)

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']

        masked_acc, correct, total = compute_masked_accuracy(
            s_codes, t_codes, lengths, encoder_stride
        )
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += total
        metrics['total_frames'] += s_codes.numel()

        unmasked_acc = compute_unmasked_accuracy(output['student_codes'], output['teacher_codes'])
        metrics['unmasked_acc'] += unmasked_acc

        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        dist = distance_matrix[s_flat, t_flat].mean().item()
        metrics['distance_loss'] += dist

        n_batches += 1

    for key in ['total_loss', 'feature_loss', 'triplet_loss', 'ce_loss',
                'masked_acc', 'unmasked_acc', 'distance_loss']:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=3, split='val'):
    """保存音檔樣本"""
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

        noisy_path = audio_dir / f'sample_{i+1}_noisy.wav'
        torchaudio.save(str(noisy_path), noisy_audio.cpu(), sample_rate)

        clean_path = audio_dir / f'sample_{i+1}_clean.wav'
        torchaudio.save(str(clean_path), clean_audio.cpu(), sample_rate)

        try:
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            student_path = audio_dir / f'sample_{i+1}_student_recon.wav'
            torchaudio.save(str(student_path), student_recon.cpu(), sample_rate)
            del student_features, student_recon
            torch.cuda.empty_cache()

            teacher_features, _, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(device))
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            teacher_path = audio_dir / f'sample_{i+1}_teacher_recon.wav'
            torchaudio.save(str(teacher_path), teacher_recon.cpu(), sample_rate)
            del teacher_features, teacher_recon
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM when saving audio sample {i+1}, skipping reconstruction")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples to {audio_dir}")


def plot_metrics(history, exp_dir):
    """繪製訓練曲線"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot([x * 100 for x in history['train_masked_acc']], label='Train Masked')
    ax.plot([x * 100 for x in history['val_masked_acc']], label='Val Masked')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Masked Token Accuracy')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(history['train_ce_loss'], label='Train')
    ax.plot(history['val_ce_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CE Loss')
    ax.set_title('Cross-Entropy Loss')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(history['train_feature_loss'], label='Train')
    ax.plot(history['val_feature_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature Loss')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(history['train_dist'], label='Train')
    ax.plot(history['val_dist'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Average Distance')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 2]
    ax.plot(history['train_triplet_loss'], label='Train')
    ax.plot(history['val_triplet_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Triplet Loss')
    ax.set_title('Triplet Loss')
    ax.legend()
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
                        choices=['all_18', 'critical_8'],
                        help='Which encoder layers to apply LoRA')

    # Loss 權重
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--triplet_weight', type=float, default=0.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=0.0)

    # 訓練參數
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--check_interval', type=int, default=100)

    # Scheduler 參數
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Encoder stride
    parser.add_argument('--encoder_stride', type=int, default=320)

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
    print(f"exp_1217: 可配置 LoRA 訓練")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)
    print(f"LoRA Config:")
    print(f"  Rank: {args.lora_rank}")
    print(f"  Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}")
    print(f"  Layers: {args.lora_layers}")
    print(f"Loss Config:")
    print(f"  Feature: {args.feature_weight}")
    print(f"  Triplet: {args.triplet_weight}")
    print(f"  CE: {args.ce_weight}")
    print("=" * 60)

    # 載入資料
    print("\n載入資料...")

    class DataConfig:
        batch_size = args.batch_size
        num_workers = args.num_workers
        pin_memory = True

    train_loader, val_loader = create_aligned_dataloaders(DataConfig())
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # 載入距離矩陣
    print("\n載入距離矩陣...")
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)
    print(f"  Distance matrix shape: {distance_matrix.shape}")

    # 創建模型
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

    # Loss Function
    loss_fn = MaskedCombinedLoss(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        ce_weight=args.ce_weight,
        encoder_stride=args.encoder_stride,
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
            print(f"  模型權重已載入")

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  Optimizer 狀態已載入")

            if args.resume_epoch is not None:
                start_epoch = args.resume_epoch
            elif 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1

            print(f"  從 Epoch {start_epoch} 繼續訓練")
        else:
            print(f"警告: checkpoint 不存在: {resume_path}")

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs - args.warmup_epochs,
            eta_min=args.lr * 0.01
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )
        print(f"\nScheduler: Cosine with {args.warmup_epochs} epoch warmup")

    # AMP
    scaler = GradScaler() if args.use_amp else None

    # 訓練歷史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_masked_acc': [], 'val_masked_acc': [],
        'train_unmasked_acc': [], 'val_unmasked_acc': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_triplet_loss': [], 'val_triplet_loss': [],
        'train_ce_loss': [], 'val_ce_loss': [],
        'train_dist': [], 'val_dist': [],
        'lr': [],
    }

    best_val_acc = 0
    best_epoch = 0

    # Resume history
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = Path(__file__).parent / resume_path

        history_path = resume_path.parent / 'history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                old_history = json.load(f)
            for key in history.keys():
                if key in old_history:
                    history[key] = old_history[key][:start_epoch - 1]
            print(f"  訓練歷史已載入 ({len(history['train_loss'])} epochs)")

            if history['val_masked_acc']:
                best_val_acc = max(history['val_masked_acc'])
                best_epoch = history['val_masked_acc'].index(best_val_acc) + 1
                print(f"  之前最佳: Epoch {best_epoch}, Masked Acc: {best_val_acc*100:.2f}%")

    # 訓練循環
    print("\n開始訓練...")
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, args.encoder_stride, scaler, args.use_amp, args.check_interval,
            args.grad_clip
        )

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        if scheduler is not None:
            scheduler.step()
            print(f"  LR: {current_lr:.2e}")

        val_metrics = validate(
            model, val_loader, loss_fn, device,
            distance_matrix, args.encoder_stride, args.use_amp
        )

        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_masked_acc'].append(train_metrics['masked_acc'])
        history['val_masked_acc'].append(val_metrics['masked_acc'])
        history['train_unmasked_acc'].append(train_metrics['unmasked_acc'])
        history['val_unmasked_acc'].append(val_metrics['unmasked_acc'])
        history['train_feature_loss'].append(train_metrics['feature_loss'])
        history['val_feature_loss'].append(val_metrics['feature_loss'])
        history['train_triplet_loss'].append(train_metrics['triplet_loss'])
        history['val_triplet_loss'].append(val_metrics['triplet_loss'])
        history['train_ce_loss'].append(train_metrics['ce_loss'])
        history['val_ce_loss'].append(val_metrics['ce_loss'])
        history['train_dist'].append(train_metrics['distance_loss'])
        history['val_dist'].append(val_metrics['distance_loss'])

        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, "
              f"Masked Acc={train_metrics['masked_acc']*100:.2f}%, "
              f"Unmasked Acc={train_metrics['unmasked_acc']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, "
              f"Masked Acc={val_metrics['masked_acc']*100:.2f}%, "
              f"Unmasked Acc={val_metrics['unmasked_acc']*100:.2f}%")
        print(f"       Distance={val_metrics['distance_loss']:.4f}")

        train_pad_ratio = 1 - train_metrics['valid_frames'] / train_metrics['total_frames']
        val_pad_ratio = 1 - val_metrics['valid_frames'] / val_metrics['total_frames']
        print(f"       Padding: Train={train_pad_ratio*100:.1f}%, Val={val_pad_ratio*100:.1f}%")

        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_masked_acc': val_metrics['masked_acc'],
                'val_unmasked_acc': val_metrics['unmasked_acc'],
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best model saved! Masked Acc: {best_val_acc*100:.2f}%")

        if epoch % 10 == 0 or epoch == 1:
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

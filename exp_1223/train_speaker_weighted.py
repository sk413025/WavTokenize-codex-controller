"""
exp_1223: Speaker-Weighted Loss Training Script

基於 exp_1219/train.py，使用 Speaker-Weighted Loss:
- 根據 speaker embedding 與 training speakers 的相似度調整 loss weight
- 對 unseen speakers 給予較低的 penalty
- 不改變 features，只在 loss 層面加入 speaker information
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

from exp_1223.data_speaker import create_speaker_aware_dataloaders
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1223.losses_speaker import SpeakerWeightedCombinedLoss
from exp_1219.losses import compute_masked_accuracy


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
    loss_fn: SpeakerWeightedCombinedLoss,
    device: str,
    epoch: int,
    distance_matrix: torch.Tensor,
    encoder_stride: int = 320,
    scaler=None,
    use_amp: bool = True,
    check_interval: int = 100,
    grad_clip: float = 1.0,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """訓練一個 epoch"""
    model.train()
    loss_fn.train()  # Important: update speaker centroid
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'cosine_loss': 0, 'cos_sim_mean': 0,
        'masked_acc': 0, 'unmasked_acc': 0, 'distance_loss': 0,
        'speaker_weight_mean': 0, 'speaker_weight_std': 0,
        'valid_frames': 0, 'total_frames': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)
        speaker_embedding = batch['speaker_embedding'].to(device)

        # 只在累積開始時清零梯度
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                # 不使用 speaker conditioning，只用原始模型
                output = model(noisy_audio, clean_audio)

                # 使用 speaker-weighted loss
                loss, loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                    speaker_embeddings=speaker_embedding,  # 傳入 speaker embedding 給 loss
                    update_speaker_centroid=True,
                )

            # 縮放 loss 以匹配累積步數
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            # 只在累積完成時更新參數
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
                speaker_embeddings=speaker_embedding,
                update_speaker_centroid=True,
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
        metrics['cosine_loss'] += loss_info.get('cosine_loss', 0)
        metrics['cos_sim_mean'] += loss_info.get('cos_sim_mean', 0)
        metrics['speaker_weight_mean'] += loss_info.get('speaker_weight_mean', 1.0)
        metrics['speaker_weight_std'] += loss_info.get('speaker_weight_std', 0)

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

        spk_w = loss_info.get('speaker_weight_mean', 1.0)
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'm_acc': f"{masked_acc*100:.1f}%",
            'spk_w': f"{spk_w:.2f}"
        })

    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    loss_fn: SpeakerWeightedCombinedLoss,
    device: str,
    distance_matrix: torch.Tensor,
    encoder_stride: int = 320,
    use_amp: bool = True,
) -> dict:
    """驗證模型"""
    model.eval()
    loss_fn.eval()  # Don't update speaker centroid during validation
    verify_model_state(model, "Validation")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'cosine_loss': 0, 'cos_sim_mean': 0,
        'masked_acc': 0, 'unmasked_acc': 0, 'distance_loss': 0,
        'speaker_weight_mean': 0, 'speaker_weight_std': 0,
        'valid_frames': 0, 'total_frames': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)
        speaker_embedding = batch['speaker_embedding'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)

            loss, loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
                speaker_embeddings=speaker_embedding,
                update_speaker_centroid=False,  # Don't update during validation
            )

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['cosine_loss'] += loss_info.get('cosine_loss', 0)
        metrics['cos_sim_mean'] += loss_info.get('cos_sim_mean', 0)
        metrics['speaker_weight_mean'] += loss_info.get('speaker_weight_mean', 1.0)
        metrics['speaker_weight_std'] += loss_info.get('speaker_weight_std', 0)

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

    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=2, split='val'):
    """保存音檔樣本 (使用正確的 WavTokenizer decode 流程)"""
    model.eval()
    audio_dir = exp_dir / 'audio_samples' / split / f'epoch_{epoch:03d}'
    audio_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 24000
    data_iter = iter(dataloader)

    torch.cuda.empty_cache()

    for i in range(num_samples):
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

        # Save original audio
        import torchaudio
        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'), noisy_audio.cpu(), sample_rate)
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'), clean_audio.cpu(), sample_rate)

        try:
            # Student reconstruction: encode with student, decode with teacher
            student_features, _, _ = model.student.feature_extractor(noisy_audio, bandwidth_id=0)
            student_recon = model.teacher.decode(student_features, bandwidth_id=torch.tensor([0]).to(device))
            if student_recon.dim() == 3:
                student_recon = student_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_student_recon.wav'), student_recon.cpu(), sample_rate)
            del student_features, student_recon
            torch.cuda.empty_cache()

            # Teacher reconstruction: encode and decode with teacher (for reference)
            teacher_features, _, _ = model.teacher.feature_extractor(clean_audio, bandwidth_id=0)
            teacher_recon = model.teacher.decode(teacher_features, bandwidth_id=torch.tensor([0]).to(device))
            if teacher_recon.dim() == 3:
                teacher_recon = teacher_recon.squeeze(1)
            torchaudio.save(str(audio_dir / f'sample_{i+1}_teacher_recon.wav'), teacher_recon.cpu(), sample_rate)
            del teacher_features, teacher_recon
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Warning: Failed to save audio sample {i+1}: {e}")
            continue

    print(f"  Saved {num_samples} {split} audio samples to {audio_dir}")


def plot_metrics(history, exp_dir):
    """繪製訓練曲線 (包含 speaker weight)"""
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
    ax.plot([x * 100 for x in history['train_masked_acc']], label='Train Masked')
    ax.plot([x * 100 for x in history['val_masked_acc']], label='Val Masked')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Masked Token Accuracy')
    ax.legend()
    ax.grid(True)

    # Speaker Weight
    ax = axes[0, 2]
    if 'train_speaker_weight' in history and history['train_speaker_weight']:
        ax.plot(history['train_speaker_weight'], label='Train')
        ax.plot(history['val_speaker_weight'], label='Val')
        ax.axhline(y=0.5, color='r', linestyle='--', label='Min Weight')
        ax.axhline(y=1.0, color='g', linestyle='--', label='Max Weight')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Speaker Weight')
    ax.set_title('Speaker Weight (Mean)')
    ax.legend()
    ax.grid(True)

    # Cosine Similarity (if available)
    ax = axes[0, 3]
    if history.get('train_cos_sim', []) and any(history['train_cos_sim']):
        ax.plot(history['train_cos_sim'], label='Train')
        ax.plot(history['val_cos_sim'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cosine Similarity')
    ax.legend()
    ax.grid(True)

    # Feature Loss
    ax = axes[1, 0]
    ax.plot(history['train_feature_loss'], label='Train')
    ax.plot(history['val_feature_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature Loss')
    ax.legend()
    ax.grid(True)

    # Triplet Loss
    ax = axes[1, 1]
    ax.plot(history['train_triplet_loss'], label='Train')
    ax.plot(history['val_triplet_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Triplet Loss')
    ax.set_title('Triplet Loss')
    ax.legend()
    ax.grid(True)

    # Distance
    ax = axes[1, 2]
    ax.plot(history['train_dist'], label='Train')
    ax.plot(history['val_dist'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Average Distance')
    ax.legend()
    ax.grid(True)

    # Val Accuracy Zoomed
    ax = axes[1, 3]
    val_acc = [x * 100 for x in history['val_masked_acc']]
    ax.plot(val_acc, label='Val Masked Acc', color='orange')
    if val_acc:
        ax.axhline(y=max(val_acc), color='g', linestyle='--', label=f'Best: {max(val_acc):.2f}%')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Val Masked Accuracy (Zoomed)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"Saved training curves to {exp_dir / 'training_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description='Speaker-Weighted Loss Training')

    # Model args
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_layers', type=str, default='all_18')

    # Loss args
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=0.0)

    # Speaker weighting args (NEW)
    parser.add_argument('--speaker_min_weight', type=float, default=0.5,
                        help='Minimum weight for unseen speakers')
    parser.add_argument('--speaker_temperature', type=float, default=1.0,
                        help='Temperature for speaker weight calculation')
    parser.add_argument('--use_speaker_weighting', action='store_true', default=True)
    parser.add_argument('--no_speaker_weighting', dest='use_speaker_weighting', action='store_false')

    # Training args
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--pin_memory', action='store_true', default=True)

    # Experiment args
    parser.add_argument('--exp_name', type=str, default='exp61_speaker_weighted')
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.output_dir is None:
        args.output_dir = f'/home/sbplab/ruizi/WavTokenize-self-supervised/exp_1223/runs/{args.exp_name}'

    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Speaker Weighting: {args.use_speaker_weighting}")
    print(f"  min_weight: {args.speaker_min_weight}")
    print(f"  temperature: {args.speaker_temperature}")
    print("=" * 60)

    # Data
    train_loader, val_loader = create_speaker_aware_dataloaders(args)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")

    # Model (使用原始 LoRA 模型，不用 speaker conditioning)
    model = TeacherStudentConfigurableLoRA(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        device=device,
    )

    # Loss (使用 speaker-weighted loss)
    loss_fn = SpeakerWeightedCombinedLoss(
        feature_weight=args.feature_weight,
        cosine_weight=args.cosine_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        ce_weight=args.ce_weight,
        speaker_min_weight=args.speaker_min_weight,
        speaker_temperature=args.speaker_temperature,
        use_speaker_weighting=args.use_speaker_weighting,
    )

    # Distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX, weights_only=True).to(device)

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
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=args.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )

    # AMP
    scaler = GradScaler() if args.use_amp else None

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = {'train': [], 'val': []}

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, scaler=scaler, use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, loss_fn, device, distance_matrix,
            use_amp=args.use_amp,
        )

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Log
        print(f"Train: Loss={train_metrics['total_loss']:.4f}, "
              f"Masked Acc={train_metrics['masked_acc']*100:.2f}%, "
              f"Spk Weight={train_metrics['speaker_weight_mean']:.3f}")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, "
              f"Masked Acc={val_metrics['masked_acc']*100:.2f}%, "
              f"Spk Weight={val_metrics['speaker_weight_mean']:.3f}")

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save best model
        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            torch.save(model.state_dict(), exp_dir / 'best_model.pt')
            print(f"  ★ New best model saved! Masked Acc: {best_val_acc*100:.2f}%")

        # Save audio samples every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            save_audio_samples(model, train_loader, device, exp_dir, epoch, split='train')
            save_audio_samples(model, val_loader, device, exp_dir, epoch, split='val')

        # Save checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
            }, exp_dir / f'checkpoint_epoch_{epoch}.pt')

        # Save history
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Masked Acc: {best_val_acc*100:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

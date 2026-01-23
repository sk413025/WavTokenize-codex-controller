#!/usr/bin/env python3
"""
Exp K v5 Continue: 從 checkpoint 繼續訓練到 500 epoch

基於 train_v5.py 修改，確保所有組件完全一致:
- Model: TeacherStudentIntermediate (rank=256, alpha=512, dropout=0.2)
- Loss: MaskedCombinedLossV2 (feature=1.0, triplet=1.0, margin=0.2)
- IntermediateSupervisionLossV5: L3(0.3), L4(1.0), L6(0.5)
- Optimizer: AdamW (weight_decay=0.1)

變更:
1. 載入已訓練的 checkpoint (epoch 300 best_model.pt)
2. 重新設定 learning rate scheduler (5e-6 → 1e-6, cosine decay)
3. Curriculum 固定在 85%
4. Intermediate weight 固定在 0.25
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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy
from exp_1226.data_curriculum import (
    create_curriculum_dataloaders,
    CurriculumDataset,
    collate_fn_curriculum
)


# ============================================================
# IntermediateSupervisionLossV5 - 與 train_v5.py 完全一致
# ============================================================
class IntermediateSupervisionLossV5(nn.Module):
    """
    V5: 動態權重中間層監督 Loss

    配置:
    - L3/model[3] (0.3): Downsample
    - L4/model[4] (1.0): ResBlock
    - L6/model[6] (0.5): Downsample
    - 全部使用 Cosine Loss
    """

    def __init__(self, layer_weights: dict = None, target_scale: float = 1.0):
        super().__init__()

        if layer_weights is None:
            layer_weights = {
                3: 0.3,   # model[3]: Downsample 1
                4: 1.0,   # model[4]: ResBlock 2
                6: 0.5,   # model[6]: Downsample 2
            }

        self.layer_weights = layer_weights
        self.layer_indices = sorted(layer_weights.keys())
        self.target_scale = target_scale

        print(f"[V5] Intermediate Supervision Config:")
        print(f"     Layers: {self.layer_indices}")
        print(f"     Weights: {layer_weights}")
        print(f"     Loss: All Cosine")

    def forward(self, student_features: dict, teacher_features: dict,
                layer_scale: float = 1.0):
        total_loss = 0.0
        layer_losses = {}

        for idx in self.layer_indices:
            if idx in student_features:
                student_feat = student_features[idx]
                teacher_feat = teacher_features[idx]
            elif str(idx) in student_features:
                student_feat = student_features[str(idx)]
                teacher_feat = teacher_features[str(idx)]
            else:
                continue

            if student_feat.shape != teacher_feat.shape:
                min_len = min(student_feat.shape[-1], teacher_feat.shape[-1])
                student_feat = student_feat[..., :min_len]
                teacher_feat = teacher_feat[..., :min_len]

            if student_feat.dim() == 3:
                B, C, T = student_feat.shape
                student_flat = student_feat.permute(0, 2, 1).reshape(-1, C)
                teacher_flat = teacher_feat.permute(0, 2, 1).reshape(-1, C)
            else:
                student_flat = student_feat
                teacher_flat = teacher_feat

            student_norm = F.normalize(student_flat, dim=-1)
            teacher_norm = F.normalize(teacher_flat * self.target_scale, dim=-1)
            cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
            loss = 1 - cos_sim

            weight = self.layer_weights[idx] * layer_scale
            total_loss = total_loss + weight * loss
            layer_losses[f'intermediate_L{idx}_loss'] = loss.item()

        layer_losses['layer_scale'] = layer_scale

        return total_loss, layer_losses


def get_trainable_params(model):
    """獲取可訓練參數 - 與 train_v5.py 一致"""
    return [p for p in model.parameters() if p.requires_grad]


def verify_model_state(model, stage: str):
    """驗證 teacher 保持 eval 模式"""
    if model.teacher.training:
        raise RuntimeError(f"[{stage}] Teacher 意外進入 train 模式!")


# ============================================================
# Training Functions - 與 train_v5.py 完全一致
# ============================================================
def train_epoch(model, dataloader, optimizer, loss_fn, intermediate_loss_fn, device, epoch,
                intermediate_weight, encoder_stride=320, scaler=None, use_amp=True, grad_clip=1.0):
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'intermediate_loss': 0,
        'intermediate_L3_loss': 0, 'intermediate_L4_loss': 0,
        'intermediate_L6_loss': 0,
        'masked_acc': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                output = model(noisy_audio, clean_audio)

                # Final output loss - 與 train_v5.py 一致的調用方式
                final_loss, final_loss_info = loss_fn(
                    student_features=output['student_encoder_out'],
                    teacher_features=output['teacher_encoder_out'],
                    teacher_codes=output['teacher_codes'],
                    codebook=output['codebook'],
                    lengths=lengths,
                )

                # Intermediate supervision loss
                inter_loss, inter_loss_info = intermediate_loss_fn(
                    student_features=output['student_intermediates'],
                    teacher_features=output['teacher_intermediates'],
                )

                total_loss = final_loss + intermediate_weight * inter_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(noisy_audio, clean_audio)

            final_loss, final_loss_info = loss_fn(
                student_features=output['student_encoder_out'],
                teacher_features=output['teacher_encoder_out'],
                teacher_codes=output['teacher_codes'],
                codebook=output['codebook'],
                lengths=lengths,
            )

            inter_loss, inter_loss_info = intermediate_loss_fn(
                student_features=output['student_intermediates'],
                teacher_features=output['teacher_intermediates'],
            )

            total_loss = final_loss + intermediate_weight * inter_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
            optimizer.step()

        # Update metrics
        metrics['total_loss'] += total_loss.item()
        metrics['feature_loss'] += final_loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += final_loss_info.get('triplet_loss', 0)
        metrics['intermediate_loss'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss

        for layer_idx in [3, 4, 6]:
            key = f'intermediate_L{layer_idx}_loss'
            if key in inter_loss_info:
                metrics[f'intermediate_L{layer_idx}_loss'] += inter_loss_info[key]

        # Compute masked accuracy - 與 train_v5.py 一致
        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        masked_acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'm_acc': f"{masked_acc*100:.2f}%",
            'inter': f"{inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss:.4f}",
            'iw': f"{intermediate_weight:.3f}"
        })

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate_epoch(model, dataloader, loss_fn, intermediate_loss_fn, device, epoch,
                   intermediate_weight, encoder_stride=320):
    model.eval()

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'intermediate_loss': 0,
        'intermediate_L3_loss': 0, 'intermediate_L4_loss': 0,
        'intermediate_L6_loss': 0,
        'masked_acc': 0,
    }
    n_batches = 0

    for batch in dataloader:
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        output = model(noisy_audio, clean_audio)

        final_loss, final_loss_info = loss_fn(
            student_features=output['student_encoder_out'],
            teacher_features=output['teacher_encoder_out'],
            teacher_codes=output['teacher_codes'],
            codebook=output['codebook'],
            lengths=lengths,
        )

        inter_loss, inter_loss_info = intermediate_loss_fn(
            student_features=output['student_intermediates'],
            teacher_features=output['teacher_intermediates'],
        )

        total_loss = final_loss + intermediate_weight * inter_loss

        metrics['total_loss'] += total_loss.item()
        metrics['feature_loss'] += final_loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += final_loss_info.get('triplet_loss', 0)
        metrics['intermediate_loss'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss

        for layer_idx in [3, 4, 6]:
            key = f'intermediate_L{layer_idx}_loss'
            if key in inter_loss_info:
                metrics[f'intermediate_L{layer_idx}_loss'] += inter_loss_info[key]

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        masked_acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc

        n_batches += 1

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


def plot_training_curves(history, save_path, start_epoch=1):
    """Plot training curves - 與 train_v5.py 一致"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(start_epoch, start_epoch + len(history['train_loss']))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    train_acc = [x * 100 for x in history['train_masked_acc']]
    val_acc = [x * 100 for x in history['val_masked_acc']]
    ax.plot(epochs, train_acc, 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, val_acc, 'r-', label='Val', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Masked Accuracy (%)')
    ax.set_title('Token Match Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Intermediate Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['val_intermediate_loss'], 'g-', label='Total', alpha=0.7)
    if 'val_intermediate_L3_loss' in history:
        ax.plot(epochs, history['val_intermediate_L3_loss'], 'b--', label='L3', alpha=0.5)
        ax.plot(epochs, history['val_intermediate_L4_loss'], 'r--', label='L4', alpha=0.5)
        ax.plot(epochs, history['val_intermediate_L6_loss'], 'm--', label='L6', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Intermediate Loss')
    ax.set_title('Intermediate Layer Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'b-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Continue Exp K v5 training')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default='exp_k_v5_continue')
    parser.add_argument('--start_epoch', type=int, default=301)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)

    # Learning rate - 繼續訓練使用較低的 LR
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # 固定參數 (繼續訓練不改變)
    parser.add_argument('--intermediate_weight', type=float, default=0.25)
    parser.add_argument('--curriculum_phase', type=float, default=0.85)

    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--save_audio_interval', type=int, default=50)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = checkpoint_dir.parent / f"{args.exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exp K v5 Continue Training")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {run_dir}")
    print(f"Epochs: {args.start_epoch} → {args.num_epochs}")
    print(f"LR: {args.lr} → {args.min_lr}")
    print(f"Intermediate Weight: {args.intermediate_weight} (fixed)")
    print(f"Curriculum: {args.curriculum_phase:.0%} (fixed)")
    print(f"{'='*60}\n")

    # Load previous history
    prev_history_path = checkpoint_dir / 'history.json'
    if prev_history_path.exists():
        with open(prev_history_path) as f:
            prev_history = json.load(f)
        print(f"Loaded previous history: {len(prev_history['train_loss'])} epochs")
        best_val_acc = max(prev_history['val_masked_acc'])
        best_epoch = prev_history['val_masked_acc'].index(best_val_acc) + 1
        print(f"Previous best: {best_val_acc*100:.3f}% @ Epoch {best_epoch}")
    else:
        prev_history = None
        best_val_acc = 0
        best_epoch = 0

    # ============================================================
    # Create model - 與 train_v5.py 完全一致
    # ============================================================
    print("\nCreating model...")
    intermediate_indices = [3, 4, 6]
    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=256,
        lora_alpha=512,
        lora_dropout=0.2,
        intermediate_indices=intermediate_indices,
        device=str(device),
    )
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully")

    # ============================================================
    # Loss functions - 與 train_v5.py 完全一致
    # ============================================================
    loss_fn = MaskedCombinedLossV2(
        feature_weight=1.0,
        triplet_weight=1.0,
        triplet_margin=0.2,
    )

    intermediate_loss_fn = IntermediateSupervisionLossV5(
        layer_weights={3: 0.3, 4: 1.0, 6: 0.5},
        target_scale=1.0
    )

    # ============================================================
    # Optimizer - 與 train_v5.py 完全一致
    # ============================================================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.1
    )

    # Learning rate scheduler - 針對繼續訓練重新設計
    total_continue_epochs = args.num_epochs - args.start_epoch + 1

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (total_continue_epochs - args.warmup_epochs)
            return max(args.min_lr / args.lr, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ============================================================
    # Data loaders - 與 train_v5.py 完全一致
    # ============================================================
    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=4,
        curriculum_mode='curriculum',
        compute_snr=False,  # 重要: 與 train_v5.py 一致，避免重新計算 SNR
    )

    # 設定固定的 curriculum phase
    if curriculum_sampler is not None:
        curriculum_sampler.current_phase = args.curriculum_phase
        curriculum_sampler._update_indices()
        print(f"Curriculum fixed at {args.curriculum_phase:.0%} ({len(curriculum_sampler.indices)} samples)")

    # Initialize history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_masked_acc': [], 'val_masked_acc': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_triplet_loss': [], 'val_triplet_loss': [],
        'train_intermediate_loss': [], 'val_intermediate_loss': [],
        'train_intermediate_L3_loss': [], 'val_intermediate_L3_loss': [],
        'train_intermediate_L4_loss': [], 'val_intermediate_L4_loss': [],
        'train_intermediate_L6_loss': [], 'val_intermediate_L6_loss': [],
        'lr': [],
    }

    # Save config
    config = vars(args)
    config['previous_best_acc'] = best_val_acc
    config['previous_best_epoch'] = best_epoch
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training
    scaler = GradScaler() if args.use_amp else None

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, intermediate_loss_fn,
            device, epoch, args.intermediate_weight,
            scaler=scaler, use_amp=args.use_amp, grad_clip=args.grad_clip
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, intermediate_loss_fn,
            device, epoch, args.intermediate_weight
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_masked_acc'].append(train_metrics['masked_acc'])
        history['val_masked_acc'].append(val_metrics['masked_acc'])
        history['train_feature_loss'].append(train_metrics['feature_loss'])
        history['val_feature_loss'].append(val_metrics['feature_loss'])
        history['train_triplet_loss'].append(train_metrics['triplet_loss'])
        history['val_triplet_loss'].append(val_metrics['triplet_loss'])
        history['train_intermediate_loss'].append(train_metrics['intermediate_loss'])
        history['val_intermediate_loss'].append(val_metrics['intermediate_loss'])
        history['train_intermediate_L3_loss'].append(train_metrics['intermediate_L3_loss'])
        history['val_intermediate_L3_loss'].append(val_metrics['intermediate_L3_loss'])
        history['train_intermediate_L4_loss'].append(train_metrics['intermediate_L4_loss'])
        history['val_intermediate_L4_loss'].append(val_metrics['intermediate_L4_loss'])
        history['train_intermediate_L6_loss'].append(train_metrics['intermediate_L6_loss'])
        history['val_intermediate_L6_loss'].append(val_metrics['intermediate_L6_loss'])
        history['lr'].append(current_lr)

        # Check for new best
        current_val_acc = val_metrics['masked_acc']
        is_best = current_val_acc > best_val_acc

        if is_best:
            best_val_acc = current_val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, run_dir / 'best_model.pt')
            print(f"  * New best Val Acc: {best_val_acc*100:.3f}%")

        # Print summary
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}, Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"  Train Acc: {train_metrics['masked_acc']*100:.3f}%, Val Acc: {val_metrics['masked_acc']*100:.3f}%")
        print(f"  LR: {current_lr:.8f}")

        # Save periodically
        if epoch % 10 == 0:
            with open(run_dir / 'history.json', 'w') as f:
                json.dump(history, f)
            plot_training_curves(history, run_dir / 'training_curves.png', start_epoch=args.start_epoch)

    # Final save
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    plot_training_curves(history, run_dir / 'training_curves.png', start_epoch=args.start_epoch)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Acc: {best_val_acc*100:.3f}% @ Epoch {best_epoch}")
    print(f"Output directory: {run_dir}")


if __name__ == '__main__':
    main()

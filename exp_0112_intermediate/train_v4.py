"""
exp_0112_intermediate: Exp K v4 - 優化版中間層監督訓練

改進重點:
1. 移除 L10 監督 (效果存疑，佔比僅 0.01%)
2. 調整權重配置: L5 提高, L6 降低
3. 加強正則化: 提高 weight_decay
4. 降低中間層總權重，避免過擬合

配置:
- L3 (0.3): low_level 輔助, Cosine Loss
- L5 (1.0): mid_level 協同 (收斂最好), Cosine Loss
- L6 (0.5): 噪音處理核心 (降低權重), Cosine Loss
- intermediate_weight: 0.5 (總權重降低)

執行:
    bash exp_0112_intermediate/run_exp_k_v4.sh
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
from exp_0112_intermediate.models import TeacherStudentIntermediate
from exp_1219.losses import MaskedCombinedLossV2, compute_masked_accuracy
from exp_1226.data_curriculum import (
    create_curriculum_dataloaders,
    CurriculumDataset,
    collate_fn_curriculum
)


class IntermediateSupervisionLossV4(nn.Module):
    """
    V4: 簡化版中間層監督 Loss
    - 移除 L10 (效果存疑)
    - 只監督 L3, L5, L6
    - 全部使用 Cosine Loss
    """

    def __init__(self, layer_weights: dict = None, target_scale: float = 1.0):
        super().__init__()

        # 預設權重配置
        if layer_weights is None:
            layer_weights = {
                3: 0.3,   # low_level 輔助
                5: 1.0,   # mid_level 協同 (收斂最好，提高權重)
                6: 0.5,   # mid_level 核心 (降低權重避免過擬合)
            }

        self.layer_weights = layer_weights
        self.layer_indices = sorted(layer_weights.keys())
        self.target_scale = target_scale

        print(f"[V4] Intermediate Supervision Config:")
        print(f"     Layers: {self.layer_indices}")
        print(f"     Weights: {layer_weights}")
        print(f"     Loss: All Cosine")

    def forward(self, student_features: dict, teacher_features: dict):
        """
        計算中間層監督 Loss

        Returns:
            total_loss: 總 Loss
            layer_losses: 各層 Loss dict
        """
        total_loss = 0.0
        layer_losses = {}

        for idx in self.layer_indices:
            key = str(idx)

            if key not in student_features or key not in teacher_features:
                continue

            student_feat = student_features[key]
            teacher_feat = teacher_features[key]

            # 確保維度匹配
            if student_feat.shape != teacher_feat.shape:
                min_len = min(student_feat.shape[1], teacher_feat.shape[1])
                student_feat = student_feat[:, :min_len, :]
                teacher_feat = teacher_feat[:, :min_len, :]

            # Cosine Loss for all layers
            student_norm = F.normalize(student_feat, dim=-1)
            teacher_norm = F.normalize(teacher_feat * self.target_scale, dim=-1)
            cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
            loss = 1 - cos_sim

            weight = self.layer_weights[idx]
            total_loss = total_loss + weight * loss
            layer_losses[f'L{idx}'] = loss.item()

        return total_loss, layer_losses


def get_exp_k_v4_config():
    """獲取 Exp K v4 的配置"""
    return {
        'intermediate_indices': [3, 5, 6],
        'layer_weights': {
            3: 0.3,   # low_level
            5: 1.0,   # mid_level (提高)
            6: 0.5,   # mid_level (降低)
        },
        'intermediate_weight': 0.5,  # 總權重降低
        'target_scale': 1.0,
    }


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


def train_epoch(model, dataloader, optimizer, loss_fn, intermediate_loss_fn, device, epoch,
                distance_matrix, intermediate_weight, encoder_stride=320, scaler=None, use_amp=True,
                check_interval=100, grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'intermediate_loss': 0,
        'intermediate_L3_loss': 0, 'intermediate_L5_loss': 0,
        'intermediate_L6_loss': 0,
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
                outputs = model(noisy_audio, clean_audio, lengths)

                student_indices = outputs['student_indices']
                teacher_indices = outputs['teacher_indices']
                student_features = outputs['student_features']
                teacher_features = outputs['teacher_features']
                student_intermediate = outputs.get('student_intermediate', {})
                teacher_intermediate = outputs.get('teacher_intermediate', {})

                frame_lengths = lengths // encoder_stride

                main_loss, loss_dict = loss_fn(
                    student_indices, teacher_indices,
                    student_features, teacher_features,
                    distance_matrix, frame_lengths
                )

                if intermediate_loss_fn is not None and intermediate_weight > 0:
                    inter_loss, inter_layer_losses = intermediate_loss_fn(
                        student_intermediate, teacher_intermediate
                    )
                    total_loss = main_loss + intermediate_weight * inter_loss
                else:
                    inter_loss = torch.tensor(0.0)
                    inter_layer_losses = {}
                    total_loss = main_loss

                total_loss = total_loss / gradient_accumulation_steps

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(noisy_audio, clean_audio, lengths)

            student_indices = outputs['student_indices']
            teacher_indices = outputs['teacher_indices']
            student_features = outputs['student_features']
            teacher_features = outputs['teacher_features']
            student_intermediate = outputs.get('student_intermediate', {})
            teacher_intermediate = outputs.get('teacher_intermediate', {})

            frame_lengths = lengths // encoder_stride

            main_loss, loss_dict = loss_fn(
                student_indices, teacher_indices,
                student_features, teacher_features,
                distance_matrix, frame_lengths
            )

            if intermediate_loss_fn is not None and intermediate_weight > 0:
                inter_loss, inter_layer_losses = intermediate_loss_fn(
                    student_intermediate, teacher_intermediate
                )
                total_loss = main_loss + intermediate_weight * inter_loss
            else:
                inter_loss = torch.tensor(0.0)
                inter_layer_losses = {}
                total_loss = main_loss

            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        # Update metrics
        metrics['total_loss'] += total_loss.item() * gradient_accumulation_steps
        metrics['feature_loss'] += loss_dict.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_dict.get('triplet_loss', 0)
        metrics['intermediate_loss'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss

        for layer in ['L3', 'L5', 'L6']:
            if layer in inter_layer_losses:
                metrics[f'intermediate_{layer}_loss'] += inter_layer_losses[layer]

        with torch.no_grad():
            masked_acc = compute_masked_accuracy(student_indices, teacher_indices, frame_lengths)
            metrics['masked_acc'] += masked_acc.item()

        if 'snr' in batch:
            metrics['avg_snr'] += batch['snr'].mean().item()

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{total_loss.item()*gradient_accumulation_steps:.4f}",
            'm_acc': f"{masked_acc.item()*100:.2f}%",
            'inter': f"{inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss:.4f}"
        })

        if batch_idx % check_interval == 0:
            verify_model_state(model, f"Epoch {epoch} Batch {batch_idx}")

    # Average metrics
    for key in metrics:
        if key not in ['valid_frames', 'total_frames']:
            metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate_epoch(model, dataloader, loss_fn, intermediate_loss_fn, device, epoch,
                   distance_matrix, intermediate_weight, encoder_stride=320):
    model.eval()

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'intermediate_loss': 0,
        'intermediate_L3_loss': 0, 'intermediate_L5_loss': 0,
        'intermediate_L6_loss': 0,
        'masked_acc': 0, 'distance': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        outputs = model(noisy_audio, clean_audio, lengths)

        student_indices = outputs['student_indices']
        teacher_indices = outputs['teacher_indices']
        student_features = outputs['student_features']
        teacher_features = outputs['teacher_features']
        student_intermediate = outputs.get('student_intermediate', {})
        teacher_intermediate = outputs.get('teacher_intermediate', {})

        frame_lengths = lengths // encoder_stride

        main_loss, loss_dict = loss_fn(
            student_indices, teacher_indices,
            student_features, teacher_features,
            distance_matrix, frame_lengths
        )

        if intermediate_loss_fn is not None and intermediate_weight > 0:
            inter_loss, inter_layer_losses = intermediate_loss_fn(
                student_intermediate, teacher_intermediate
            )
            total_loss = main_loss + intermediate_weight * inter_loss
        else:
            inter_loss = torch.tensor(0.0)
            inter_layer_losses = {}
            total_loss = main_loss

        metrics['total_loss'] += total_loss.item()
        metrics['feature_loss'] += loss_dict.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_dict.get('triplet_loss', 0)
        metrics['intermediate_loss'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss

        for layer in ['L3', 'L5', 'L6']:
            if layer in inter_layer_losses:
                metrics[f'intermediate_{layer}_loss'] += inter_layer_losses[layer]

        masked_acc = compute_masked_accuracy(student_indices, teacher_indices, frame_lengths)
        metrics['masked_acc'] += masked_acc.item()

        # Distance
        B = student_indices.shape[0]
        total_dist = 0
        for b in range(B):
            valid_len = frame_lengths[b].item()
            for t in range(valid_len):
                s_idx = student_indices[b, t].item()
                t_idx = teacher_indices[b, t].item()
                total_dist += distance_matrix[s_idx, t_idx].item()
        metrics['distance'] += total_dist / frame_lengths.sum().item()

        n_batches += 1

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def save_audio_samples(model, dataloader, device, exp_dir, epoch, num_samples=2, split='val'):
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

        # Save noisy and clean
        torchaudio.save(str(audio_dir / f'sample_{i+1}_noisy.wav'), noisy_audio.cpu(), sample_rate)
        torchaudio.save(str(audio_dir / f'sample_{i+1}_clean.wav'), clean_audio.cpu(), sample_rate)

        # Save student reconstruction
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


def plot_training_curves_v4(history, save_path):
    """繪製 V4 訓練曲線"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Exp K v4: 3-Layer Intermediate Supervision (L3+L5+L6)', fontsize=14)

    epochs = range(1, len(history['train_loss']) + 1)

    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Masked Token Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [x*100 for x in history['train_masked_acc']], 'b-', label='Train')
    ax.plot(epochs, [x*100 for x in history['val_masked_acc']], 'r-', label='Val')
    ax.set_title('Masked Token Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True)

    # Feature Loss
    ax = axes[0, 2]
    ax.plot(epochs, history['train_feature_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_feature_loss'], 'r-', label='Val')
    ax.set_title('Feature Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Triplet Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_triplet_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_triplet_loss'], 'r-', label='Val')
    ax.set_title('Triplet Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Intermediate Supervision Loss (Total)
    ax = axes[1, 1]
    ax.plot(epochs, history['train_intermediate_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_intermediate_loss'], 'r-', label='Val')
    ax.set_title('Intermediate Supervision Loss (Total)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Train-Val Accuracy Gap
    ax = axes[1, 2]
    gap = [(t - v) * 100 for t, v in zip(history['train_masked_acc'], history['val_masked_acc'])]
    ax.plot(epochs, gap, 'g-')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title('Train-Val Accuracy Gap')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (%)')
    ax.grid(True)

    # Per-Layer Intermediate Loss (Train)
    ax = axes[2, 0]
    ax.plot(epochs, history['train_intermediate_L3_loss'], 'b-', label='L3 (Cosine, w=0.3)')
    ax.plot(epochs, history['train_intermediate_L5_loss'], 'g--', label='L5 (Cosine, w=1.0)')
    ax.plot(epochs, history['train_intermediate_L6_loss'], 'r-.', label='L6 (Cosine, w=0.5)')
    ax.set_title('V4: Per-Layer Intermediate Loss (Train)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    # Learning Rate
    ax = axes[2, 1]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'b-')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True)

    # Curriculum Phase
    ax = axes[2, 2]
    if 'curriculum_phase' in history:
        ax.plot(epochs, history['curriculum_phase'], 'orange')
    ax.set_title('Curriculum Phase')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Max Noise Ratio')
    ax.set_ylim(0, 1.1)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Exp K v4: Optimized Intermediate Layer Supervision')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp_k_v4')
    parser.add_argument('--output_dir', type=str, default=None)

    # LoRA Config
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)

    # Learning Rate
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # V4: Intermediate Supervision (3 layers, no L10)
    parser.add_argument('--intermediate_weight', type=float, default=0.5,
                        help='Total intermediate loss weight (reduced from 1.0)')
    parser.add_argument('--intermediate_L3_weight', type=float, default=0.3,
                        help='Weight for L3 (low_level)')
    parser.add_argument('--intermediate_L5_weight', type=float, default=1.0,
                        help='Weight for L5 (mid_level, best convergence)')
    parser.add_argument('--intermediate_L6_weight', type=float, default=0.5,
                        help='Weight for L6 (mid_level, reduced)')
    parser.add_argument('--target_scale', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=0.0)

    # Training
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay (increased for regularization)')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    # Curriculum
    parser.add_argument('--curriculum_start', type=float, default=0.3)
    parser.add_argument('--curriculum_end', type=float, default=1.0)
    parser.add_argument('--curriculum_epochs', type=int, default=100)

    # AMP
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')

    # Audio samples
    parser.add_argument('--save_audio_interval', type=int, default=50,
                        help='Save audio samples every N epochs')

    # Random seed
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = Path('exp_0112_intermediate/runs') / f"{args.exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['v4_info'] = {
        'description': 'Optimized intermediate supervision',
        'changes': [
            'Removed L10 (ineffective)',
            'L5 weight increased to 1.0 (best convergence)',
            'L6 weight reduced to 0.5 (avoid overfitting)',
            'Total intermediate_weight reduced to 0.5',
            'weight_decay increased to 0.1',
        ]
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exp K v4: Optimized Intermediate Layer Supervision")
    print(f"{'='*60}")
    print(f"Output: {run_dir}")
    print(f"Config: L3(w={args.intermediate_L3_weight}) + L5(w={args.intermediate_L5_weight}) + L6(w={args.intermediate_L6_weight})")
    print(f"Total intermediate_weight: {args.intermediate_weight}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"{'='*60}\n")

    # Load distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX).to(device)

    # Create model (V4: only L3, L5, L6)
    intermediate_indices = [3, 5, 6]

    model = TeacherStudentIntermediate(
        config_path=WAVTOK_CONFIG,
        ckpt_path=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers='all',
        intermediate_indices=intermediate_indices,
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler (cosine with warmup)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)
            return max(args.min_lr / args.lr, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create data loaders
    train_loader, val_loader = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=4,
        initial_noise_ratio=args.curriculum_start,
    )

    # Create loss functions
    loss_fn = MaskedCombinedLossV2(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
    )

    # V4 intermediate loss function
    layer_weights = {
        3: args.intermediate_L3_weight,
        5: args.intermediate_L5_weight,
        6: args.intermediate_L6_weight,
    }
    intermediate_loss_fn = IntermediateSupervisionLossV4(
        layer_weights=layer_weights,
        target_scale=args.target_scale,
    )

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_masked_acc': [], 'val_masked_acc': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_triplet_loss': [], 'val_triplet_loss': [],
        'train_intermediate_loss': [], 'val_intermediate_loss': [],
        'train_intermediate_L3_loss': [], 'val_intermediate_L3_loss': [],
        'train_intermediate_L5_loss': [], 'val_intermediate_L5_loss': [],
        'train_intermediate_L6_loss': [], 'val_intermediate_L6_loss': [],
        'train_dist': [], 'val_dist': [],
        'train_avg_snr': [],
        'curriculum_phase': [],
        'lr': [],
    }

    best_val_acc = 0
    best_epoch = 0

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        # Update curriculum
        if epoch <= args.curriculum_epochs:
            progress = epoch / args.curriculum_epochs
            noise_ratio = args.curriculum_start + progress * (args.curriculum_end - args.curriculum_start)
        else:
            noise_ratio = args.curriculum_end

        train_loader.dataset.set_noise_ratio(noise_ratio)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs} (Curriculum: {int(noise_ratio*100)}%)")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, intermediate_loss_fn,
            device, epoch, distance_matrix, args.intermediate_weight,
            scaler=scaler, use_amp=args.use_amp, grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, intermediate_loss_fn,
            device, epoch, distance_matrix, args.intermediate_weight
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

        for layer in ['L3', 'L5', 'L6']:
            key = f'intermediate_{layer}_loss'
            history[f'train_{key}'].append(train_metrics.get(key, 0))
            history[f'val_{key}'].append(val_metrics.get(key, 0))

        history['val_dist'].append(val_metrics['distance'])
        history['train_avg_snr'].append(train_metrics['avg_snr'])
        history['curriculum_phase'].append(noise_ratio)
        history['lr'].append(current_lr)

        # Print summary
        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, Acc={train_metrics['masked_acc']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, Acc={val_metrics['masked_acc']*100:.2f}%")
        print(f"Inter: L3={val_metrics.get('intermediate_L3_loss', 0):.4f}, "
              f"L5={val_metrics.get('intermediate_L5_loss', 0):.4f}, "
              f"L6={val_metrics.get('intermediate_L6_loss', 0):.4f}")
        print(f"LR: {current_lr:.6f}")

        # Save best model
        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, run_dir / 'best_model.pt')
            print(f"★ New best Val Acc: {best_val_acc*100:.3f}%")

        # Save audio samples
        if epoch % args.save_audio_interval == 0 or epoch == 1:
            print(f"\nSaving audio samples...")
            save_audio_samples(model, val_loader, device, run_dir, epoch, num_samples=2, split='val')
            save_audio_samples(model, train_loader, device, run_dir, epoch, num_samples=2, split='train')

        # Save history and plot
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if epoch % 10 == 0 or epoch == args.num_epochs:
            plot_training_curves_v4(history, run_dir / 'training_curves.png')

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Acc: {best_val_acc*100:.3f}% @ Epoch {best_epoch}")
    print(f"Output directory: {run_dir}")

    # Final plot
    plot_training_curves_v4(history, run_dir / 'training_curves.png')


if __name__ == '__main__':
    main()

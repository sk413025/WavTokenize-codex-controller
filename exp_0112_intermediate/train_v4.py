"""
exp_0112_intermediate: Exp K v4 - 優化版中間層監督訓練

改進重點:
1. 移除 L10 監督 (效果存疑)
2. 修正監督層: model[5] 是 ELU，改為 model[4] (ResBlock2)
3. 加強正則化: 提高 weight_decay
4. 降低中間層總權重，避免過擬合

encoder.model 結構:
  model[3]: SConv1d (Downsample 1) - 監督目標
  model[4]: SEANetResnetBlock (ResBlock 2) - 監督目標 (修正)
  model[5]: ELU (激活函數，監督無效!)
  model[6]: SConv1d (Downsample 2) - 監督目標

配置:
- L3/model[3] (0.3): Downsample, Cosine Loss
- L4/model[4] (1.0): ResBlock (修正), Cosine Loss
- L6/model[6] (0.5): Downsample, Cosine Loss
- intermediate_weight: 0.5

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
    - 監督 L3, L4, L6 (修正: L5 是 ELU，改為 L4 ResBlock)
    - 全部使用 Cosine Loss
    """

    def __init__(self, layer_weights: dict = None, target_scale: float = 1.0):
        super().__init__()

        # 預設權重配置 (修正: 5->4)
        if layer_weights is None:
            layer_weights = {
                3: 0.3,   # model[3]: Downsample 1
                4: 1.0,   # model[4]: ResBlock 2 (修正，原本誤用 model[5] ELU)
                6: 0.5,   # model[6]: Downsample 2
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
            layer_losses: 各層 Loss dict (with keys like 'intermediate_L3_loss')
        """
        total_loss = 0.0
        layer_losses = {}

        for idx in self.layer_indices:
            # Support both int and str keys
            if idx in student_features:
                student_feat = student_features[idx]
                teacher_feat = teacher_features[idx]
            elif str(idx) in student_features:
                student_feat = student_features[str(idx)]
                teacher_feat = teacher_features[str(idx)]
            else:
                continue

            # 確保維度匹配 (B, T, C) or (B, C, T)
            if student_feat.shape != teacher_feat.shape:
                # Assume (B, C, T) format, match on T dimension
                min_len = min(student_feat.shape[-1], teacher_feat.shape[-1])
                student_feat = student_feat[..., :min_len]
                teacher_feat = teacher_feat[..., :min_len]

            # Flatten to (B*T, C) for cosine similarity
            if student_feat.dim() == 3:
                B, C, T = student_feat.shape
                student_flat = student_feat.permute(0, 2, 1).reshape(-1, C)
                teacher_flat = teacher_feat.permute(0, 2, 1).reshape(-1, C)
            else:
                student_flat = student_feat
                teacher_flat = teacher_feat

            # Cosine Loss for all layers
            student_norm = F.normalize(student_flat, dim=-1)
            teacher_norm = F.normalize(teacher_flat * self.target_scale, dim=-1)
            cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
            loss = 1 - cos_sim

            weight = self.layer_weights[idx]
            total_loss = total_loss + weight * loss
            layer_losses[f'intermediate_L{idx}_loss'] = loss.item()

        return total_loss, layer_losses


def get_exp_k_v4_config():
    """獲取 Exp K v4 的配置"""
    return {
        'intermediate_indices': [3, 4, 6],  # 修正: 5->4
        'layer_weights': {
            3: 0.3,   # model[3]: Downsample 1
            4: 1.0,   # model[4]: ResBlock 2 (修正)
            6: 0.5,   # model[6]: Downsample 2
        },
        'intermediate_weight': 0.5,
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
        'intermediate_L3_loss': 0, 'intermediate_L4_loss': 0,
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
                # Model forward (compatible with TeacherStudentIntermediate)
                output = model(noisy_audio, clean_audio)

                # Final output loss
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

                # Combined loss
                total_loss = final_loss + intermediate_weight * inter_loss
                total_loss = total_loss / gradient_accumulation_steps

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
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
            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        # Update metrics
        metrics['total_loss'] += total_loss.item() * gradient_accumulation_steps
        metrics['feature_loss'] += final_loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += final_loss_info.get('triplet_loss', 0)
        metrics['intermediate_loss'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss

        for layer_idx in [3, 4, 6]:  # V4: 修正為 [3, 4, 6]
            key = f'intermediate_L{layer_idx}_loss'
            if key in inter_loss_info:
                metrics[f'intermediate_L{layer_idx}_loss'] += inter_loss_info[key]

        # Compute masked accuracy
        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        masked_acc, correct, total_frames = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc
        metrics['valid_frames'] += correct
        metrics['total_frames'] += total_frames

        if 'snr' in batch:
            metrics['avg_snr'] += batch['snr'].mean().item()

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{total_loss.item()*gradient_accumulation_steps:.4f}",
            'm_acc': f"{masked_acc*100:.2f}%",
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
        'intermediate_L3_loss': 0, 'intermediate_L4_loss': 0,
        'intermediate_L6_loss': 0,
        'masked_acc': 0, 'distance': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        output = model(noisy_audio, clean_audio)

        # Final output loss
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

        # Combined loss
        total_loss = final_loss + intermediate_weight * inter_loss

        metrics['total_loss'] += total_loss.item()
        metrics['feature_loss'] += final_loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += final_loss_info.get('triplet_loss', 0)
        metrics['intermediate_loss'] += inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss

        for layer_idx in [3, 4, 6]:  # V4: 修正為 [3, 4, 6]
            key = f'intermediate_L{layer_idx}_loss'
            if key in inter_loss_info:
                metrics[f'intermediate_L{layer_idx}_loss'] += inter_loss_info[key]

        # Compute masked accuracy
        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        masked_acc, correct, total_frames = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc

        # Distance (compute if needed)
        metrics['distance'] += 0  # Placeholder, can compute later if needed

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
    fig.suptitle('Exp K v4: 3-Layer Intermediate Supervision (L3+L4+L6)', fontsize=14)

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
    ax.plot(epochs, history['train_intermediate_L4_loss'], 'g--', label='L4 (Cosine, w=1.0)')
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
    parser.add_argument('--intermediate_L4_weight', type=float, default=1.0,
                        help='Weight for L4 (ResBlock2, noise sensitive)')
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
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

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
            'Fixed: model[5] is ELU, changed to model[4] (ResBlock2)',
            'L4 weight = 1.0 (noise sensitive core)',
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
    print(f"Config: L3(w={args.intermediate_L3_weight}) + L4(w={args.intermediate_L4_weight}) + L6(w={args.intermediate_L6_weight})")
    print(f"Total intermediate_weight: {args.intermediate_weight}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"{'='*60}\n")

    # Load distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX).to(device)

    # Create model (V4: L3, L4, L6)
    # 修正: model[5] 是 ELU，改為 model[4] (ResBlock2)
    intermediate_indices = [3, 4, 6]

    model = TeacherStudentIntermediate(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=4,
        curriculum_mode='curriculum',
        initial_phase=args.curriculum_start,
        phase_increment=0.1,
        compute_snr=False,
    )

    # Create loss functions
    loss_fn = MaskedCombinedLossV2(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
    )

    # V4 intermediate loss function (修正: 5->4)
    layer_weights = {
        3: args.intermediate_L3_weight,
        4: args.intermediate_L4_weight,  # model[4] ResBlock2
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
        'train_intermediate_L4_loss': [], 'val_intermediate_L4_loss': [],
        'train_intermediate_L6_loss': [], 'val_intermediate_L6_loss': [],
        'train_dist': [], 'val_dist': [],
        'train_avg_snr': [],
        'curriculum_phase': [],
        'lr': [],
    }

    best_val_acc = 0
    best_epoch = 0

    # Curriculum phase advance interval (similar to v2)
    phase_advance_epochs = 30

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        # Update curriculum phase
        if epoch > 1 and epoch % phase_advance_epochs == 0:
            curriculum_sampler.advance_phase()

        current_phase = curriculum_sampler.current_phase

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs} (Curriculum Phase: {current_phase:.0%})")
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

        for layer_idx in [3, 4, 6]:  # V4: 修正為 [3, 4, 6]
            key = f'intermediate_L{layer_idx}_loss'
            history[f'train_{key}'].append(train_metrics.get(key, 0))
            history[f'val_{key}'].append(val_metrics.get(key, 0))

        history['val_dist'].append(val_metrics['distance'])
        history['train_avg_snr'].append(train_metrics['avg_snr'])
        history['curriculum_phase'].append(current_phase)
        history['lr'].append(current_lr)

        # Print summary
        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, Acc={train_metrics['masked_acc']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, Acc={val_metrics['masked_acc']*100:.2f}%")
        print(f"Inter: L3={val_metrics.get('intermediate_L3_loss', 0):.4f}, "
              f"L4={val_metrics.get('intermediate_L4_loss', 0):.4f}, "
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

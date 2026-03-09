#!/usr/bin/env python3
"""
exp_0121_token_guided: Token-Guided LoRA Training

基於 Exp K 架構，整合 token 分析結果:
- Loss: MSE + Triplet + 中間層監督 (Exp K 驗證有效)
- Token-Guided: 對高錯誤率 token 可選加權
- Noise-Type Aware: 對難噪音類型可選加權

執行:
    bash run_exp_a.sh  # 標準 Exp K 架構
    bash run_exp_b.sh  # + Token-Weighted
    bash run_exp_c.sh  # + Layer-Selective LoRA
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

from exp_0121_token_guided.models import TeacherStudentTokenGuided


# ============================================================
# 路徑配置 (沿用 Exp K - 使用 exp_1201.config)
# ============================================================
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE


# ============================================================
# Loss Functions (沿用 exp_1219)
# ============================================================
from exp_1219.losses import (
    MaskedCombinedLossV2,
    MaskedCosineLoss,
    compute_masked_accuracy,
    create_length_mask,
)


class IntermediateSupervisionLoss(nn.Module):
    """
    中間層監督 Loss (沿用 Exp K)

    使用 Cosine Loss 監督 L3, L4, L6
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

        print(f"[Intermediate Supervision]")
        print(f"  Layers: {self.layer_indices}")
        print(f"  Weights: {layer_weights}")
        print(f"  Loss: Cosine")

    def forward(self, student_features: dict, teacher_features: dict,
                layer_scale: float = 1.0):
        """
        計算中間層監督 Loss

        Args:
            student_features: Student 中間層特徵
            teacher_features: Teacher 中間層特徵
            layer_scale: 動態縮放因子

        Returns:
            total_loss: 總 Loss
            layer_losses: 各層 Loss dict
        """
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

            # 確保維度匹配
            if student_feat.shape != teacher_feat.shape:
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

            # Cosine Loss
            student_norm = F.normalize(student_flat, dim=-1)
            teacher_norm = F.normalize(teacher_flat * self.target_scale, dim=-1)
            cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
            loss = 1 - cos_sim

            # 動態縮放
            weight = self.layer_weights[idx] * layer_scale
            total_loss = total_loss + weight * loss
            layer_losses[f'intermediate_L{idx}_loss'] = loss.item()

        layer_losses['layer_scale'] = layer_scale

        return total_loss, layer_losses


# ============================================================
# Token-Weighted Extension (可選)
# ============================================================
class TokenWeightedWrapper(nn.Module):
    """
    Token-Weighted Loss 包裝器

    對高錯誤率 token 給更高 loss 權重
    """

    def __init__(self, base_loss_fn, error_rates_path: str = None,
                 high_error_threshold: float = 0.7,
                 high_error_multiplier: float = 2.0):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.high_error_threshold = high_error_threshold
        self.high_error_multiplier = high_error_multiplier

        # 載入 error rates
        if error_rates_path and Path(error_rates_path).exists():
            self.register_buffer('error_rates', torch.load(error_rates_path))
            print(f"[Token-Weighted] Loaded error rates from {error_rates_path}")
            high_count = (self.error_rates > high_error_threshold).sum().item()
            print(f"  High error tokens (>{high_error_threshold}): {high_count}")
        else:
            self.error_rates = None
            print("[Token-Weighted] No error rates loaded, using uniform weights")

    def forward(self, student_features, teacher_features, teacher_codes,
                codebook, lengths, logits=None):
        # 使用基礎 loss function
        total_loss, loss_dict = self.base_loss_fn(
            student_features, teacher_features, teacher_codes,
            codebook, lengths, logits
        )

        # 如果有 error rates，計算加權因子
        if self.error_rates is not None:
            # teacher_codes: (B, T) or (1, B, T)
            if teacher_codes.dim() == 3:
                codes = teacher_codes[0]
            else:
                codes = teacher_codes

            # 確保 error_rates 在正確的 device
            error_rates_device = self.error_rates.to(codes.device)
            # 獲取每個位置的 error rate
            error_rates = error_rates_device[codes.long()]  # (B, T)

            # 計算平均權重因子
            weights = torch.where(
                error_rates > self.high_error_threshold,
                torch.full_like(error_rates, self.high_error_multiplier),
                torch.ones_like(error_rates)
            )
            avg_weight = weights.mean().item()

            # 調整總 loss
            total_loss = total_loss * avg_weight
            loss_dict['token_weight_factor'] = avg_weight

        return total_loss, loss_dict


# ============================================================
# Dataset (沿用 curriculum 架構)
# ============================================================
from exp_1226.data_curriculum import (
    create_curriculum_dataloaders,
    CurriculumDataset,
    collate_fn_curriculum
)


# ============================================================
# Training Functions
# ============================================================
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


def train_epoch(model, dataloader, optimizer, loss_fn, intermediate_loss_fn, device, epoch,
                intermediate_weight, encoder_stride=320, scaler=None,
                use_amp=True, grad_clip=1.0, gradient_accumulation_steps=1):
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

    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        lengths = batch['lengths'].to(device)

        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(enabled=use_amp):
                output = model(noisy_audio, clean_audio)

                # Final output loss (MSE + Triplet)
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

        for layer_idx in [3, 4, 6]:
            key = f'intermediate_L{layer_idx}_loss'
            if key in inter_loss_info:
                metrics[f'intermediate_L{layer_idx}_loss'] += inter_loss_info[key]

        # Compute masked accuracy
        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        masked_acc, correct, total_frames = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)
        metrics['masked_acc'] += masked_acc

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{total_loss.item()*gradient_accumulation_steps:.4f}",
            'm_acc': f"{masked_acc*100:.2f}%",
            'inter': f"{inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss:.4f}",
        })

    # Average metrics
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

    for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
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


def save_wav(filepath, waveform, sample_rate):
    """使用 scipy 保存 wav 檔案，避免 torchaudio 的 torchcodec 依賴問題"""
    import scipy.io.wavfile as wavfile
    import numpy as np

    # 確保是 numpy array
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()

    # 確保是 (samples,) 或 (channels, samples)
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)  # (1, T) -> (T,)

    # 轉換為 int16
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform_int16 = (waveform * 32767).astype(np.int16)

    wavfile.write(str(filepath), sample_rate, waveform_int16)


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

        save_wav(audio_dir / f'sample_{i+1}_noisy.wav', noisy_audio, sample_rate)
        save_wav(audio_dir / f'sample_{i+1}_clean.wav', clean_audio, sample_rate)

        try:
            output = model(noisy_audio, clean_audio)
            student_recon = model.decode(output['student_encoder_out'])
            if student_recon.dim() == 1:
                student_recon = student_recon.unsqueeze(0)
            save_wav(audio_dir / f'sample_{i+1}_student_recon.wav', student_recon, sample_rate)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM when saving audio sample {i+1}, skipping")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"  Saved {min(num_samples, len(dataloader))} {split} audio samples")


def plot_training_curves(history, save_path):
    """繪製訓練曲線"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Token-Guided LoRA Training (Exp K Architecture)', fontsize=14)

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
    ax.set_title('Feature Loss (MSE)')
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

    # Intermediate Supervision Loss
    ax = axes[1, 1]
    ax.plot(epochs, history['train_intermediate_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_intermediate_loss'], 'r-', label='Val')
    ax.set_title('Intermediate Supervision Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    # Learning Rate
    ax = axes[1, 2]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'b-')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Token-Guided LoRA Training (Exp K Architecture)')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='exp_token_guided')
    parser.add_argument('--output_dir', type=str, default=None)

    # LoRA config (沿用 Exp K 預設值)
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_target_layers', type=str, nargs='*', default=None,
                        help='Layer patterns for selective LoRA')

    # Loss weights (沿用 Exp K)
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--intermediate_weight', type=float, default=0.5)

    # Token-weighted (可選)
    parser.add_argument('--use_token_weighted', action='store_true', default=False)
    parser.add_argument('--token_error_rates_path', type=str, default=None)
    parser.add_argument('--high_error_threshold', type=float, default=0.7)
    parser.add_argument('--high_error_multiplier', type=float, default=2.0)

    # Training
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

    # Curriculum (沿用 Exp K)
    parser.add_argument('--curriculum_start', type=float, default=0.3)
    parser.add_argument('--curriculum_end', type=float, default=0.85)
    parser.add_argument('--curriculum_epochs', type=int, default=200)

    # AMP
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')

    # Saving
    parser.add_argument('--save_audio_interval', type=int, default=50)

    # Seed
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
        run_dir = Path(__file__).parent / 'runs' / f"{args.exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['architecture'] = 'Based on Exp K: MSE + Triplet + Intermediate Supervision'
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Token-Guided LoRA Training (Exp K Architecture)")
    print(f"{'='*60}")
    print(f"Experiment: {args.exp_name}")
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    if args.lora_target_layers:
        print(f"Target layers: {args.lora_target_layers}")
    else:
        print(f"Target layers: ALL 18 layers")
    print(f"Loss: MSE({args.feature_weight}) + Triplet({args.triplet_weight}) + Intermediate({args.intermediate_weight})")
    if args.use_token_weighted:
        print(f"Token-Weighted: threshold={args.high_error_threshold}, multiplier={args.high_error_multiplier}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    # distance_matrix removed - not used in current loss functions

    # Create model
    model = TeacherStudentTokenGuided(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_layer_patterns=args.lora_target_layers,
        intermediate_indices=[3, 4, 6],  # Exp K 預設
        device=str(device),
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)
            return max(args.min_lr / args.learning_rate, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create data loaders (沿用 curriculum)
    num_phases = int((args.curriculum_end - args.curriculum_start) / 0.05) + 1
    phase_advance_epochs = max(1, args.curriculum_epochs // num_phases)
    phase_increment = (args.curriculum_end - args.curriculum_start) / num_phases

    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        train_cache_path=TRAIN_CACHE,
        val_cache_path=VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=4,
        curriculum_mode='curriculum',
        initial_phase=args.curriculum_start,
        phase_increment=phase_increment,
        compute_snr=False,
    )

    # Create loss functions
    base_loss_fn = MaskedCombinedLossV2(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
    )

    # 如果使用 token-weighted，包裝 loss function
    if args.use_token_weighted:
        loss_fn = TokenWeightedWrapper(
            base_loss_fn,
            error_rates_path=args.token_error_rates_path,
            high_error_threshold=args.high_error_threshold,
            high_error_multiplier=args.high_error_multiplier,
        )
    else:
        loss_fn = base_loss_fn

    # Intermediate supervision loss
    intermediate_loss_fn = IntermediateSupervisionLoss(
        layer_weights={3: 0.3, 4: 1.0, 6: 0.5}
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
        'curriculum_phase': [],
        'lr': [],
    }

    best_val_acc = 0
    best_epoch = 0

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        # Update curriculum phase
        if epoch > 1 and epoch <= args.curriculum_epochs and epoch % phase_advance_epochs == 0:
            if curriculum_sampler.current_phase < args.curriculum_end:
                curriculum_sampler.advance_phase()
                if curriculum_sampler.current_phase > args.curriculum_end:
                    curriculum_sampler.current_phase = args.curriculum_end

        current_phase = curriculum_sampler.current_phase

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Curriculum Phase: {current_phase:.0%}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, intermediate_loss_fn,
            device, epoch, args.intermediate_weight,
            scaler=scaler, use_amp=args.use_amp, grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps
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
        history['curriculum_phase'].append(current_phase)
        history['lr'].append(current_lr)

        # Print summary
        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, Acc={train_metrics['masked_acc']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, Acc={val_metrics['masked_acc']*100:.2f}%")
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
            print(f"* New best Val Acc: {best_val_acc*100:.3f}%")

        # Save audio samples
        if epoch % args.save_audio_interval == 0 or epoch == 1:
            print(f"\nSaving audio samples...")
            save_audio_samples(model, val_loader, device, run_dir, epoch, num_samples=2, split='val')

        # Save history and plot
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if epoch % 10 == 0 or epoch == args.num_epochs:
            plot_training_curves(history, run_dir / 'training_curves.png')

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Acc: {best_val_acc*100:.3f}% @ Epoch {best_epoch}")
    print(f"Output directory: {run_dir}")

    # Final plot
    plot_training_curves(history, run_dir / 'training_curves.png')


if __name__ == '__main__':
    main()

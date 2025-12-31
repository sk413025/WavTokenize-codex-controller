"""
Exp75: Progressive Loss Schedule

核心改進：
- Phase 1 (1-100): 只訓練 Feature Loss (連續空間)
- Phase 2 (101-200): 加入 Soft Token Loss (漸進增加)
- Phase 3 (201-300): Soft Token 為主
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
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1231.losses import SoftTokenLoss, ProgressiveLossScheduler
from exp_1231.utils import plot_metrics, save_audio_samples


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_masked_accuracy(student_codes, teacher_codes, lengths, encoder_stride=320):
    batch_size = student_codes.shape[0]
    total_correct = 0
    total_tokens = 0
    for i in range(batch_size):
        if lengths is not None and i < len(lengths):
            valid_frames = lengths[i].item() // encoder_stride
            valid_frames = min(valid_frames, student_codes.shape[1])
        else:
            valid_frames = student_codes.shape[1]
        if valid_frames > 0:
            s = student_codes[i, :valid_frames]
            t = teacher_codes[i, :valid_frames]
            total_correct += (s == t).sum().item()
            total_tokens += valid_frames
    return total_correct / total_tokens if total_tokens > 0 else 0, total_correct, total_tokens


class ProgressiveLoss(nn.Module):
    """支援動態權重的 Loss"""

    def __init__(self, triplet_margin=0.2, soft_token_temperature=1.0,
                 vq_commitment_weight=0.1, vq_distortion_weight=0.1):
        super().__init__()
        self.triplet_margin = triplet_margin
        self.vq_commitment_weight = vq_commitment_weight
        self.vq_distortion_weight = vq_distortion_weight
        self.soft_token_loss = SoftTokenLoss(temperature=soft_token_temperature)

    def forward(self, student_features, teacher_features, teacher_codes, codebook,
                lengths, encoder_stride, weights):
        """
        weights: dict with 'feature_weight', 'triplet_weight', 'soft_token_weight'
        """
        B, T, D = student_features.shape
        device = student_features.device

        # Mask
        if lengths is not None:
            mask = torch.zeros(B, T, device=device)
            for i in range(B):
                valid_frames = lengths[i].item() // encoder_stride
                valid_frames = min(valid_frames, T)
                mask[i, :valid_frames] = 1.0
        else:
            mask = torch.ones(B, T, device=device)

        total_loss = 0
        info = {'phase': weights.get('phase', 0)}

        # Feature Loss
        if weights['feature_weight'] > 0:
            feature_diff = (student_features - teacher_features).pow(2).mean(dim=-1)
            feature_loss = (feature_diff * mask).sum() / mask.sum()
            total_loss += weights['feature_weight'] * feature_loss
            info['feature_loss'] = feature_loss.item()

        # Triplet Loss
        if weights['triplet_weight'] > 0:
            pos_dist = (student_features - teacher_features).pow(2).sum(dim=-1).sqrt()
            neg_teacher = torch.roll(teacher_features, shifts=1, dims=0)
            neg_dist = (student_features - neg_teacher).pow(2).sum(dim=-1).sqrt()
            triplet = F.relu(pos_dist - neg_dist + self.triplet_margin)
            triplet_loss = (triplet * mask).sum() / mask.sum()
            total_loss += weights['triplet_weight'] * triplet_loss
            info['triplet_loss'] = triplet_loss.item()

        # Soft Token Loss
        if weights['soft_token_weight'] > 0:
            soft_loss, soft_info = self.soft_token_loss(
                student_features, teacher_features, codebook, lengths, encoder_stride
            )
            total_loss += weights['soft_token_weight'] * soft_loss
            info['soft_token_loss'] = soft_loss.item()
            info['soft_token_accuracy'] = soft_info.get('soft_token_accuracy', 0)

        # VQ Losses
        if self.vq_commitment_weight > 0:
            student_codes = self._get_codes(student_features, codebook)
            quantized = codebook[student_codes]
            commitment_loss = (student_features - quantized.detach()).pow(2).mean()
            total_loss += self.vq_commitment_weight * commitment_loss
            info['vq_commitment_loss'] = commitment_loss.item()

        info['total_loss'] = total_loss.item()
        return total_loss, info

    def _get_codes(self, features, codebook):
        B, T, D = features.shape
        flat = features.reshape(-1, D)
        distances = torch.cdist(flat, codebook, p=2)
        codes = distances.argmin(dim=-1)
        return codes.reshape(B, T)


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, weights,
                encoder_stride=320, scaler=None, use_amp=True,
                grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()
    metrics = {'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
               'soft_token_loss': 0, 'masked_acc': 0}
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (Phase {weights['phase']})", leave=False)
    for batch_idx, batch in enumerate(pbar):
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
                encoder_stride=encoder_stride,
                weights=weights,
            )
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)

        metrics['total_loss'] += loss.item() * gradient_accumulation_steps
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['soft_token_loss'] += loss_info.get('soft_token_loss', 0)
        metrics['masked_acc'] += acc
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item()*gradient_accumulation_steps:.4f}",
                          'acc': f"{acc*100:.2f}%"})

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, weights, encoder_stride=320, use_amp=True):
    model.eval()
    metrics = {'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
               'soft_token_loss': 0, 'masked_acc': 0}
    n_batches = 0

    for batch in dataloader:
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
                encoder_stride=encoder_stride,
                weights=weights,
            )

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)

        metrics['total_loss'] += loss.item()
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['soft_token_loss'] += loss_info.get('soft_token_loss', 0)
        metrics['masked_acc'] += acc
        n_batches += 1

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Exp75: Progressive Loss Schedule')
    parser.add_argument('--exp_name', type=str, default='exp75_progressive')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_layers', type=str, default='all_18')

    # Loss
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--soft_token_temperature', type=float, default=1.0)
    parser.add_argument('--vq_commitment_weight', type=float, default=0.1)
    parser.add_argument('--vq_distortion_weight', type=float, default=0.1)

    # Progressive Schedule
    parser.add_argument('--phase1_ratio', type=float, default=0.33)
    parser.add_argument('--phase2_ratio', type=float, default=0.33)

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
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    parser.add_argument('--encoder_stride', type=int, default=320)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = Path(__file__).parent / 'runs' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Exp75: Progressive Loss Schedule")
    print("=" * 60)
    print(f"Phase 1 (Epoch 1-{int(args.num_epochs*args.phase1_ratio)}): Feature only")
    print(f"Phase 2 (Epoch {int(args.num_epochs*args.phase1_ratio)+1}-{int(args.num_epochs*(args.phase1_ratio+args.phase2_ratio))}): Feature + Soft Token")
    print(f"Phase 3 (Epoch {int(args.num_epochs*(args.phase1_ratio+args.phase2_ratio))+1}-{args.num_epochs}): Soft Token dominant")
    print("=" * 60)

    # Data
    from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn
    from torch.utils.data import DataLoader
    train_dataset = AlignedNoisyCleanPairDataset(TRAIN_CACHE)
    val_dataset = AlignedNoisyCleanPairDataset(VAL_CACHE)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=aligned_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=aligned_collate_fn)

    # Model
    model = TeacherStudentConfigurableLoRA(
        wavtok_config=str(WAVTOK_CONFIG), wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_layers=args.lora_layers, device=device,
    )

    # Loss with progressive weights
    loss_fn = ProgressiveLoss(
        triplet_margin=args.triplet_margin,
        soft_token_temperature=args.soft_token_temperature,
        vq_commitment_weight=args.vq_commitment_weight,
        vq_distortion_weight=args.vq_distortion_weight,
    )

    # Scheduler for loss weights
    loss_scheduler = ProgressiveLossScheduler(
        total_epochs=args.num_epochs,
        phase1_ratio=args.phase1_ratio,
        phase2_ratio=args.phase2_ratio,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = None
    if args.use_scheduler:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=args.use_amp)

    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'phase': [], 'feature_weight': [], 'soft_token_weight': [], 'lr': []}

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    print("\n開始訓練...")
    for epoch in range(1, args.num_epochs + 1):
        # 取得當前 epoch 的 loss 權重
        weights = loss_scheduler.get_weights(epoch)
        print(f"\nEpoch {epoch}/{args.num_epochs} (Phase {weights['phase']})")
        print(f"  Weights: feature={weights['feature_weight']:.2f}, "
              f"triplet={weights['triplet_weight']:.2f}, "
              f"soft_token={weights['soft_token_weight']:.2f}")

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch,
                                    weights, args.encoder_stride, scaler, args.use_amp,
                                    args.grad_clip, args.gradient_accumulation_steps)
        val_metrics = validate(model, val_loader, loss_fn, device, weights,
                               args.encoder_stride, args.use_amp)

        if lr_scheduler:
            lr_scheduler.step()
            history['lr'].append(optimizer.param_groups[0]['lr'])

        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_acc'].append(train_metrics['masked_acc'])
        history['val_acc'].append(val_metrics['masked_acc'])
        history['phase'].append(weights['phase'])
        history['feature_weight'].append(weights['feature_weight'])
        history['soft_token_weight'].append(weights['soft_token_weight'])

        print(f"Train: Loss={train_metrics['total_loss']:.4f}, Acc={train_metrics['masked_acc']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, Acc={val_metrics['masked_acc']*100:.2f}%")

        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_masked_acc': val_metrics['masked_acc'],
                'config': config
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best: {best_val_acc*100:.2f}%")
            # 保存最佳模型的音檔樣本
            save_audio_samples(model, train_loader, val_loader, device, exp_dir, epoch)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # 繪製訓練曲線
        plot_metrics(history, exp_dir, exp_type='progressive')

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_masked_acc': val_metrics['masked_acc'],
        'config': config
    }, exp_dir / 'last_model.pt')

    print(f"\nBest Val Acc: {best_val_acc*100:.2f}% @ Epoch {best_epoch}")


if __name__ == '__main__':
    main()

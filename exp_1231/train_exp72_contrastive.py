"""
Exp72: Contrastive Token Loss (InfoNCE)

核心改進：
- 用對比學習的方式訓練
- 讓 student feature 靠近正確的 code，遠離錯誤的
- Hard Negative Mining 選擇最有挑戰的負樣本
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
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, TRAIN_CACHE, VAL_CACHE
from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1231.losses import CombinedLossExp72
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


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                encoder_stride=320, scaler=None, use_amp=True,
                grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()
    metrics = {'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
               'contrastive_loss': 0, 'contrastive_accuracy': 0, 'masked_acc': 0}
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
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
        metrics['contrastive_loss'] += loss_info.get('contrastive_loss', 0)
        metrics['contrastive_accuracy'] += loss_info.get('contrastive_accuracy', 0)
        metrics['masked_acc'] += acc
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item()*gradient_accumulation_steps:.4f}",
                          'acc': f"{acc*100:.2f}%"})

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, encoder_stride=320, use_amp=True):
    model.eval()
    metrics = {'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
               'contrastive_loss': 0, 'contrastive_accuracy': 0, 'masked_acc': 0}
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
            )

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)

        metrics['total_loss'] += loss.item()
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['contrastive_loss'] += loss_info.get('contrastive_loss', 0)
        metrics['contrastive_accuracy'] += loss_info.get('contrastive_accuracy', 0)
        metrics['masked_acc'] += acc
        n_batches += 1

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Exp72: Contrastive Token Loss')
    parser.add_argument('--exp_name', type=str, default='exp72_contrastive')
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_layers', type=str, default='all_18')
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--contrastive_weight', type=float, default=0.5)
    parser.add_argument('--contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--num_negatives', type=int, default=16)
    parser.add_argument('--hard_negative_mining', action='store_true', default=True)
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
    print(f"Exp72: Contrastive Token Loss (InfoNCE)")
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

    # Loss
    loss_fn = CombinedLossExp72(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        contrastive_weight=args.contrastive_weight,
        triplet_margin=args.triplet_margin,
        contrastive_temperature=args.contrastive_temperature,
        num_negatives=args.num_negatives,
        hard_negative_mining=args.hard_negative_mining,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_scheduler:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=args.use_amp)

    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'train_contrastive_acc': [], 'val_contrastive_acc': [], 'lr': []}

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch,
                                    args.encoder_stride, scaler, args.use_amp, args.grad_clip,
                                    args.gradient_accumulation_steps)
        val_metrics = validate(model, val_loader, loss_fn, device, args.encoder_stride, args.use_amp)

        if scheduler:
            scheduler.step()
            history['lr'].append(optimizer.param_groups[0]['lr'])

        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_acc'].append(train_metrics['masked_acc'])
        history['val_acc'].append(val_metrics['masked_acc'])
        history['train_contrastive_acc'].append(train_metrics['contrastive_accuracy'])
        history['val_contrastive_acc'].append(val_metrics['contrastive_accuracy'])

        print(f"Train: Loss={train_metrics['total_loss']:.4f}, Acc={train_metrics['masked_acc']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, Acc={val_metrics['masked_acc']*100:.2f}%")

        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_masked_acc': val_metrics['masked_acc'], 'config': config},
                       exp_dir / 'best_model.pt')
            print(f"  ★ New best: {best_val_acc*100:.2f}%")
            # 保存最佳模型的音檔樣本
            save_audio_samples(model, train_loader, val_loader, device, exp_dir, epoch)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # 繪製訓練曲線
        plot_metrics(history, exp_dir, exp_type='contrastive')

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_masked_acc': val_metrics['masked_acc'], 'config': config},
               exp_dir / 'last_model.pt')

    print(f"\nBest Val Acc: {best_val_acc*100:.2f}% @ Epoch {best_epoch}")


if __name__ == '__main__':
    main()

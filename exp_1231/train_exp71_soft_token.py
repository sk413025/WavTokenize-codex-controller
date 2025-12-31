"""
Exp71: Soft Token Loss (KL Divergence)

核心改進：
- 不只監督 argmax token，監督整個 logits 分布
- 使用 KL Divergence 讓 student 分布接近 teacher 分布
- 梯度更平滑，學習更穩定

基於 Exp67 (Curriculum + VQ-Aware) 的架構
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

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX, TRAIN_CACHE, VAL_CACHE
from exp_1217.models import TeacherStudentConfigurableLoRA
from exp_1231.losses import CombinedLossExp71
from exp_1231.utils import save_audio_samples


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_masked_accuracy(student_codes, teacher_codes, lengths, encoder_stride=320):
    """計算有效區域的 token 準確率"""
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

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return accuracy, total_correct, total_tokens


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                encoder_stride=320, scaler=None, use_amp=True,
                grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'soft_token_loss': 0, 'soft_token_accuracy': 0,
        'vq_commitment_loss': 0, 'vq_distortion_loss': 0,
        'masked_acc': 0,
    }
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

        # 計算 token accuracy
        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        acc, _, _ = compute_masked_accuracy(s_codes, t_codes, lengths, encoder_stride)

        # 累積指標
        metrics['total_loss'] += loss.item() * gradient_accumulation_steps
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['soft_token_loss'] += loss_info.get('soft_token_loss', 0)
        metrics['soft_token_accuracy'] += loss_info.get('soft_token_accuracy', 0)
        metrics['vq_commitment_loss'] += loss_info.get('vq_commitment_loss', 0)
        metrics['vq_distortion_loss'] += loss_info.get('vq_distortion_loss', 0)
        metrics['masked_acc'] += acc
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item()*gradient_accumulation_steps:.4f}",
            'acc': f"{acc*100:.2f}%",
            'soft_acc': f"{loss_info.get('soft_token_accuracy', 0)*100:.2f}%",
        })

    # 平均
    for k in metrics:
        metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, encoder_stride=320, use_amp=True):
    model.eval()

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0,
        'soft_token_loss': 0, 'soft_token_accuracy': 0,
        'vq_commitment_loss': 0, 'vq_distortion_loss': 0,
        'masked_acc': 0,
    }
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
        metrics['soft_token_loss'] += loss_info.get('soft_token_loss', 0)
        metrics['soft_token_accuracy'] += loss_info.get('soft_token_accuracy', 0)
        metrics['vq_commitment_loss'] += loss_info.get('vq_commitment_loss', 0)
        metrics['vq_distortion_loss'] += loss_info.get('vq_distortion_loss', 0)
        metrics['masked_acc'] += acc
        n_batches += 1

    for k in metrics:
        metrics[k] /= n_batches

    return metrics


def plot_metrics(history, exp_dir):
    """繪製訓練曲線"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Token Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].set_title('Token Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Soft Token Accuracy
    if 'train_soft_acc' in history:
        axes[0, 2].plot(history['train_soft_acc'], label='Train')
        axes[0, 2].plot(history['val_soft_acc'], label='Val')
        axes[0, 2].set_title('Soft Token Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

    # Feature Loss
    axes[1, 0].plot(history['train_feature_loss'], label='Train')
    axes[1, 0].plot(history['val_feature_loss'], label='Val')
    axes[1, 0].set_title('Feature Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Soft Token Loss
    if 'train_soft_token_loss' in history:
        axes[1, 1].plot(history['train_soft_token_loss'], label='Train')
        axes[1, 1].plot(history['val_soft_token_loss'], label='Val')
        axes[1, 1].set_title('Soft Token Loss (KLD)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    # Learning Rate
    if 'lr' in history:
        axes[1, 2].plot(history['lr'])
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Exp71: Soft Token Loss')

    # 實驗設定
    parser.add_argument('--exp_name', type=str, default='exp71_soft_token')

    # LoRA 參數
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--lora_layers', type=str, default='all_18')

    # Loss 參數
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--soft_token_weight', type=float, default=1.0)
    parser.add_argument('--soft_token_temperature', type=float, default=1.0)
    parser.add_argument('--vq_commitment_weight', type=float, default=0.1)
    parser.add_argument('--vq_distortion_weight', type=float, default=0.1)

    # 訓練參數
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
    print(f"Using device: {device}")

    # 實驗目錄
    exp_dir = Path(__file__).parent / 'runs' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Exp71: Soft Token Loss (KL Divergence)")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)
    print(f"Soft Token Config:")
    print(f"  Weight: {args.soft_token_weight}")
    print(f"  Temperature: {args.soft_token_temperature}")
    print("=" * 60)

    # 載入資料
    print("\n載入資料...")
    from exp_1212.data_aligned import AlignedNoisyCleanPairDataset, aligned_collate_fn
    from torch.utils.data import DataLoader

    train_dataset = AlignedNoisyCleanPairDataset(TRAIN_CACHE)
    val_dataset = AlignedNoisyCleanPairDataset(VAL_CACHE)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=aligned_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=aligned_collate_fn
    )
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    # 載入模型
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

    # Loss function
    loss_fn = CombinedLossExp71(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        soft_token_weight=args.soft_token_weight,
        triplet_margin=args.triplet_margin,
        soft_token_temperature=args.soft_token_temperature,
        vq_commitment_weight=args.vq_commitment_weight,
        vq_distortion_weight=args.vq_distortion_weight,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=args.use_amp)

    # 保存配置
    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 訓練歷史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_soft_token_loss': [], 'val_soft_token_loss': [],
        'train_soft_acc': [], 'val_soft_acc': [],
        'lr': [],
    }

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    print("\n開始訓練...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # 訓練
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            args.encoder_stride, scaler, args.use_amp,
            args.grad_clip, args.gradient_accumulation_steps
        )

        # 驗證
        val_metrics = validate(
            model, val_loader, loss_fn, device, args.encoder_stride, args.use_amp
        )

        # 更新 scheduler
        if scheduler:
            scheduler.step()
            history['lr'].append(optimizer.param_groups[0]['lr'])

        # 記錄歷史
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_acc'].append(train_metrics['masked_acc'])
        history['val_acc'].append(val_metrics['masked_acc'])
        history['train_feature_loss'].append(train_metrics['feature_loss'])
        history['val_feature_loss'].append(val_metrics['feature_loss'])
        history['train_soft_token_loss'].append(train_metrics['soft_token_loss'])
        history['val_soft_token_loss'].append(val_metrics['soft_token_loss'])
        history['train_soft_acc'].append(train_metrics['soft_token_accuracy'])
        history['val_soft_acc'].append(val_metrics['soft_token_accuracy'])

        # 打印結果
        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, "
              f"Acc={train_metrics['masked_acc']*100:.2f}%, "
              f"SoftAcc={train_metrics['soft_token_accuracy']*100:.2f}%")
        print(f"Val:   Loss={val_metrics['total_loss']:.4f}, "
              f"Acc={val_metrics['masked_acc']*100:.2f}%, "
              f"SoftAcc={val_metrics['soft_token_accuracy']*100:.2f}%")

        # 保存最佳模型
        if val_metrics['masked_acc'] > best_val_acc:
            best_val_acc = val_metrics['masked_acc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_masked_acc': val_metrics['masked_acc'],
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best model saved! Acc: {best_val_acc*100:.2f}%")
            # 保存最佳模型的音檔樣本
            save_audio_samples(model, train_loader, val_loader, device, exp_dir, epoch)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # 繪製曲線
        plot_metrics(history, exp_dir)

        # 保存歷史
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # 保存最終模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_masked_acc': val_metrics['masked_acc'],
        'config': config,
    }, exp_dir / 'last_model.pt')

    print("\n" + "=" * 60)
    print("訓練完成!")
    print(f"Best Val Acc: {best_val_acc*100:.2f}% @ Epoch {best_epoch}")
    print("=" * 60)


if __name__ == '__main__':
    main()

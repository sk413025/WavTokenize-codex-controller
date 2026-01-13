"""
exp_test: 淺層 LoRA 容量瓶頸測試

核心問題:
- 降噪擾動在中層 (L5-L8) 最大
- 但 LoRA 訓練後深層變化最大
- 懷疑: LoRA 容量不足

實驗設計:
- 只訓練 L0-L4 (5 層)，Loss 監督 L4 輸出
- 完全凍結 L5-L17 (13 層)
- 測試不同 LoRA rank (256/512/1024)

執行:
    # Rank 256
    python exp_test/train.py --exp_name shallow_r256 --lora_rank 256 --lora_alpha 512

    # Rank 512
    python exp_test/train.py --exp_name shallow_r512 --lora_rank 512 --lora_alpha 1024

    # Rank 1024
    python exp_test/train.py --exp_name shallow_r1024 --lora_rank 1024 --lora_alpha 2048

預期:
- 如果 rank↑ → loss↓↓ → 容量不足
- 如果 rank↑ → loss 無明顯改善 → 問題不在容量
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
from exp_test.models import TeacherStudentShallowOnly, ShallowMSELoss
from exp_1226.data_curriculum import (
    create_curriculum_dataloaders,
    CurriculumDataset,
    collate_fn_curriculum
)


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


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                encoder_stride=320, scaler=None, use_amp=True,
                check_interval=100, grad_clip=1.0, gradient_accumulation_steps=1):
    model.train()
    verify_model_state(model, f"Epoch {epoch} 開始")

    metrics = {
        'total_loss': 0,
        'mse_loss': 0,
        'cos_sim': 0,
        'avg_snr': 0,
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
                output = model(noisy_audio, clean_audio)

                loss, loss_info = loss_fn(
                    student_l4_out=output['student_l4_out'],
                    teacher_l4_out=output['teacher_l4_out'],
                )

            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(noisy_audio, clean_audio)

            loss, loss_info = loss_fn(
                student_l4_out=output['student_l4_out'],
                teacher_l4_out=output['teacher_l4_out'],
            )

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)
                optimizer.step()

        if (batch_idx + 1) % check_interval == 0:
            verify_model_state(model, f"Epoch {epoch} Batch {batch_idx + 1}")

        metrics['total_loss'] += loss.item()
        metrics['mse_loss'] += loss_info.get('mse_loss', 0)
        metrics['cos_sim'] += loss_info.get('cos_sim', 0)

        if 'snr' in batch:
            metrics['avg_snr'] += batch['snr'].mean().item()

        n_batches += 1
        pbar.set_postfix({
            'loss': f"{loss.item():.6f}",
            'cos_sim': f"{loss_info.get('cos_sim', 0):.4f}",
        })

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, use_amp=True):
    model.eval()
    verify_model_state(model, "Validation")

    metrics = {
        'total_loss': 0,
        'mse_loss': 0,
        'cos_sim': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)

            loss, loss_info = loss_fn(
                student_l4_out=output['student_l4_out'],
                teacher_l4_out=output['teacher_l4_out'],
            )

        metrics['total_loss'] += loss.item()
        metrics['mse_loss'] += loss_info.get('mse_loss', 0)
        metrics['cos_sim'] += loss_info.get('cos_sim', 0)

        n_batches += 1

    for key in metrics:
        metrics[key] /= n_batches

    return metrics


def plot_metrics(history, exp_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # MSE Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('L4 MSE Loss')
    ax.legend()
    ax.grid(True)

    # Cosine Similarity
    ax = axes[0, 1]
    ax.plot(history['train_cos_sim'], label='Train')
    ax.plot(history['val_cos_sim'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('L4 Cosine Similarity')
    ax.legend()
    ax.grid(True)

    # Train-Val Gap
    ax = axes[1, 0]
    gap = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
    ax.plot(gap, color='red')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap')
    ax.set_title('Train-Val Loss Gap')
    ax.grid(True)

    # Learning Rate
    ax = axes[1, 1]
    ax.plot(history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate')
    ax.grid(True)

    plt.suptitle(f'Shallow LoRA Capacity Test (rank={history.get("lora_rank", "?")})', fontsize=14)
    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='exp_test: 淺層 LoRA 容量瓶頸測試')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='shallow_capacity_test')
    parser.add_argument('--output_dir', type=str, default=None)

    # LoRA Config (主要測試參數)
    parser.add_argument('--lora_rank', type=int, default=256,
                        help='LoRA rank: 256/512/1024')
    parser.add_argument('--lora_alpha', type=int, default=512,
                        help='LoRA alpha (usually 2x rank)')
    parser.add_argument('--lora_dropout', type=float, default=0.2)

    # Learning Rate
    parser.add_argument('--lr', type=float, default=1e-4)

    # Curriculum Learning (可選)
    parser.add_argument('--curriculum_mode', type=str, default='curriculum',
                        choices=['curriculum', 'anti_curriculum'])
    parser.add_argument('--initial_phase', type=float, default=0.3)
    parser.add_argument('--phase_increment', type=float, default=0.1)
    parser.add_argument('--phase_advance_epochs', type=int, default=30)

    # Training
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='150 epochs to see capacity trend')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--check_interval', type=int, default=100)

    # Scheduler
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Gradient Accumulation
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--encoder_stride', type=int, default=320)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = Path(__file__).parent / 'runs' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config['experiment_type'] = 'Shallow LoRA Capacity Test'
    config['description'] = f'L0-L4 only, rank={args.lora_rank}'

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("exp_test: 淺層 LoRA 容量瓶頸測試")
    print("=" * 70)
    print(f"\n核心問題: LoRA 容量是否是淺層學習不足的原因?")
    print(f"\n實驗設計:")
    print(f"  - 只訓練 L0-L4 (5 層)")
    print(f"  - 凍結 L5-L17 (13 層)")
    print(f"  - Loss: L4 輸出 MSE")
    print(f"\n測試參數:")
    print(f"  ★ LoRA Rank: {args.lora_rank}")
    print(f"  ★ LoRA Alpha: {args.lora_alpha}")
    print(f"  ★ Scaling: {args.lora_alpha / args.lora_rank:.2f}")
    print(f"  - LR: {args.lr}")
    print(f"  - Epochs: {args.num_epochs}")
    print("=" * 70)

    # Load data
    print("\n載入資料 (with curriculum)...")
    train_loader, val_loader, curriculum_sampler = create_curriculum_dataloaders(
        TRAIN_CACHE, VAL_CACHE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        curriculum_mode=args.curriculum_mode,
        initial_phase=args.initial_phase,
        phase_increment=args.phase_increment,
        compute_snr=False,
    )
    print(f"  Initial curriculum: {len(curriculum_sampler)} samples ({args.initial_phase:.0%})")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print("\n創建模型...")
    model = TeacherStudentShallowOnly(
        wavtok_config=str(WAVTOK_CONFIG),
        wavtok_ckpt=str(WAVTOK_CKPT),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")

    # Loss Function
    loss_fn = ShallowMSELoss(reduction='mean')

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )

    scaler = GradScaler() if args.use_amp else None

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cos_sim': [], 'val_cos_sim': [],
        'train_avg_snr': [],
        'curriculum_phase': [],
        'lr': [],
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
    }

    best_val_loss = float('inf')
    best_epoch = 0

    # Training loop
    print("\n" + "=" * 70)
    print("開始訓練...")
    print("=" * 70)

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs} | Rank={args.lora_rank} | Curriculum: {curriculum_sampler.current_phase:.0%}")
        print(f"{'='*60}")

        if epoch > 1 and (epoch - 1) % args.phase_advance_epochs == 0:
            curriculum_sampler.advance_phase()

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            args.encoder_stride, scaler, args.use_amp,
            args.check_interval, args.grad_clip, args.gradient_accumulation_steps
        )

        # Record LR
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            scheduler.step()

        val_metrics = validate(
            model, val_loader, loss_fn, device, args.use_amp
        )

        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_cos_sim'].append(train_metrics['cos_sim'])
        history['val_cos_sim'].append(val_metrics['cos_sim'])
        history['train_avg_snr'].append(train_metrics['avg_snr'])
        history['curriculum_phase'].append(curriculum_sampler.current_phase)

        train_val_gap = train_metrics['total_loss'] - val_metrics['total_loss']

        print(f"\nTrain: Loss={train_metrics['total_loss']:.6f}, CosSim={train_metrics['cos_sim']:.4f}")
        print(f"Val:   Loss={val_metrics['total_loss']:.6f}, CosSim={val_metrics['cos_sim']:.4f}")
        print(f"Gap:   {train_val_gap:.6f}")

        # Check codebook integrity
        try:
            cb_check = model.check_codebook_integrity(raise_error=True)
        except Exception as e:
            print(f"ERROR: {e}")
            break

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
            }, exp_dir / 'best_model.pt')
            print(f"  ★ New best! Val Loss: {best_val_loss:.6f}")

        # Save history and plot
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if epoch % 10 == 0 or epoch == 1:
            plot_metrics(history, exp_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("訓練完成!")
    print("=" * 70)
    print(f"\n容量測試結果 (Rank={args.lora_rank}):")
    print(f"  Best Val Loss: {best_val_loss:.6f} @ Epoch {best_epoch}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
    print(f"  Final CosSim: {history['val_cos_sim'][-1]:.4f}")
    print(f"\n結果保存於: {exp_dir}")

    # Final plots
    plot_metrics(history, exp_dir)

    # Save final model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['total_loss'],
        'config': config,
    }, exp_dir / 'final_model.pt')

    # Summary for comparison
    summary = {
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_cos_sim': history['val_cos_sim'][-1],
        'trainable_params': trainable_params,
    }

    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("請與其他 rank 的結果比較:")
    print("  - shallow_r256: rank=256")
    print("  - shallow_r512: rank=512")
    print("  - shallow_r1024: rank=1024")
    print("\n如果 loss 隨 rank 增加而明顯降低 → 容量是瓶頸")
    print("如果 loss 無明顯改善 → 問題可能在其他地方 (e.g., 梯度流)")
    print("=" * 70)


if __name__ == '__main__':
    main()

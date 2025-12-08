"""
exp_1205: 訓練腳本

三種新的 Loss 策略：
- exp13: Linear + CE (方案 A)
- exp14: Margin-based Loss (方案 B)
- exp15: Hard Negative Mining + CE (方案 C)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp_1205.model import TeacherStudentModel
from exp_1205.data import NoisyCleanPairDataset, collate_fn
from exp_1205.losses import LinearCELoss, MarginLoss, HardNegativeCELoss, CombinedLoss
from exp_1205.config import WAVTOK_CONFIG, WAVTOK_CKPT


def parse_args():
    parser = argparse.ArgumentParser(description='exp_1205: New Loss Strategies')

    # Experiment
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)

    # Model
    parser.add_argument('--lora_rank', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=256)

    # Loss type
    parser.add_argument('--loss_type', type=str, default='linear_ce',
                        choices=['linear_ce', 'margin', 'hard_neg', 'combined'])

    # Linear CE params
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    # Margin Loss params
    parser.add_argument('--margin', type=float, default=0.5)

    # Hard Negative params
    parser.add_argument('--hard_neg_k', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)

    # Combined Loss weights
    parser.add_argument('--use_linear_ce', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_margin', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_hard_neg', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_mse', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--linear_ce_weight', type=float, default=1.0)
    parser.add_argument('--margin_weight', type=float, default=0.5)
    parser.add_argument('--hard_neg_weight', type=float, default=0.5)
    parser.add_argument('--mse_weight', type=float, default=0.1)

    # Training
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)

    # Paths
    parser.add_argument('--train_cache', type=str,
                        default='/home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/data_with_distances/train_cache_with_distances.pt')
    parser.add_argument('--val_cache', type=str,
                        default='/home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/data_with_distances/val_cache_with_distances.pt')

    # Logging
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=10)

    return parser.parse_args()


def create_loss_fn(args, device):
    """根據參數創建 Loss 函數"""
    if args.loss_type == 'linear_ce':
        loss_fn = LinearCELoss(
            embed_dim=512,
            vocab_size=4096,
            label_smoothing=args.label_smoothing
        )
    elif args.loss_type == 'margin':
        loss_fn = MarginLoss(margin=args.margin)
    elif args.loss_type == 'hard_neg':
        loss_fn = HardNegativeCELoss(
            k=args.hard_neg_k,
            temperature=args.temperature
        )
    elif args.loss_type == 'combined':
        loss_fn = CombinedLoss(
            use_linear_ce=args.use_linear_ce,
            use_margin=args.use_margin,
            use_hard_neg=args.use_hard_neg,
            use_mse=args.use_mse,
            linear_ce_weight=args.linear_ce_weight,
            margin_weight=args.margin_weight,
            hard_neg_weight=args.hard_neg_weight,
            mse_weight=args.mse_weight,
            embed_dim=512,
            vocab_size=4096,
            margin=args.margin,
            hard_neg_k=args.hard_neg_k,
            temperature=args.temperature,
            label_smoothing=args.label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    return loss_fn.to(device)


def train_epoch(model, loss_fn, train_loader, optimizer, device, epoch, args, codebook):
    """訓練一個 epoch"""
    model.train()

    # 如果 loss_fn 有可訓練參數，也設為 train 模式
    if hasattr(loss_fn, 'train'):
        loss_fn.train()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model.forward_with_emb(noisy_audio, clean_audio, compute_vq_features=False)
        student_emb = output['student_emb']
        teacher_codes = output['teacher_codes']

        # Compute loss
        loss_results = loss_fn(student_emb, teacher_codes, codebook)
        loss = loss_results['loss']

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)

        # 如果 loss_fn 有可訓練參數，也更新
        if hasattr(loss_fn, 'get_trainable_parameters'):
            for p in loss_fn.get_trainable_parameters():
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_acc += loss_results.get('token_accuracy', 0).item() if isinstance(loss_results.get('token_accuracy', 0), torch.Tensor) else loss_results.get('token_accuracy', 0)
        num_batches += 1

        # Update progress bar
        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{loss_results.get('token_accuracy', 0):.2%}" if isinstance(loss_results.get('token_accuracy', 0), (int, float)) else f"{loss_results.get('token_accuracy', 0).item():.2%}"
            })

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches
    }


@torch.no_grad()
def validate(model, loss_fn, val_loader, device, codebook):
    """驗證"""
    model.eval()
    if hasattr(loss_fn, 'eval'):
        loss_fn.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_top5 = 0.0
    total_top10 = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc='Validating'):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        output = model.forward_with_emb(noisy_audio, clean_audio, compute_vq_features=False)
        student_emb = output['student_emb']
        teacher_codes = output['teacher_codes']

        loss_results = loss_fn(student_emb, teacher_codes, codebook)

        total_loss += loss_results['loss'].item()

        acc = loss_results.get('token_accuracy', 0)
        total_acc += acc.item() if isinstance(acc, torch.Tensor) else acc

        top5 = loss_results.get('top5_accuracy', 0)
        total_top5 += top5.item() if isinstance(top5, torch.Tensor) else top5

        top10 = loss_results.get('top10_accuracy', 0)
        total_top10 += top10.item() if isinstance(top10, torch.Tensor) else top10

        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'top5_accuracy': total_top5 / num_batches,
        'top10_accuracy': total_top10 / num_batches,
    }


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create experiment directory
    exp_dir = Path(__file__).parent / 'experiments' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"exp_1205: {args.exp_name}")
    print(f"Loss type: {args.loss_type}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    model = model.to(device)

    # Get codebook
    codebook = model.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed.detach()
    print(f"Codebook shape: {codebook.shape}")

    # Create loss function
    print(f"\nCreating {args.loss_type} loss...")
    loss_fn = create_loss_fn(args, device)

    # Create optimizer
    params_to_optimize = list(model.get_trainable_parameters())
    if hasattr(loss_fn, 'get_trainable_parameters'):
        params_to_optimize.extend(list(loss_fn.get_trainable_parameters()))

    optimizer = optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    print(f"Total trainable parameters: {sum(p.numel() for p in params_to_optimize)}")

    # Load data
    print("\nLoading data...")
    train_dataset = NoisyCleanPairDataset(args.train_cache)
    val_dataset = NoisyCleanPairDataset(args.val_cache)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top5': [],
        'val_top10': [],
    }

    best_val_acc = 0.0

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_results = train_epoch(
            model, loss_fn, train_loader, optimizer, device, epoch, args, codebook
        )

        # Validate
        val_results = validate(model, loss_fn, val_loader, device, codebook)

        # Log
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_results['loss']:.4f}, Acc: {train_results['accuracy']:.2%}")
        print(f"  Val Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.2%}, "
              f"Top-5: {val_results['top5_accuracy']:.2%}, Top-10: {val_results['top10_accuracy']:.2%}")

        # Update history
        history['train_loss'].append(train_results['loss'])
        history['train_acc'].append(train_results['accuracy'])
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        history['val_top5'].append(val_results['top5_accuracy'])
        history['val_top10'].append(val_results['top10_accuracy'])

        # Save history
        with open(exp_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Save best model
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict() if hasattr(loss_fn, 'state_dict') else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config,
            }, ckpt_dir / 'best.pt')
            print(f"  * New best model saved! Val Acc: {best_val_acc:.2%}")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict() if hasattr(loss_fn, 'state_dict') else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_results['accuracy'],
                'config': config,
            }, ckpt_dir / f'epoch_{epoch}.pt')

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict() if hasattr(loss_fn, 'state_dict') else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_results['accuracy'],
            'config': config,
        }, ckpt_dir / 'latest.pt')

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best Val Acc: {best_val_acc:.2%}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

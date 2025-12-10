"""
exp_1209: 方案 D - Adapter + 方向性 Loss 訓練腳本

使用 DenoiseAdapter 修正 encoder 輸出，搭配 Triplet Loss

架構:
    Noisy Audio → Encoder(凍結) → Adapter(訓練) → VQ → tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import sys
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1201.data import create_dataloaders
from exp_1201.config import WAVTOK_CONFIG, WAVTOK_CKPT, DISTANCE_MATRIX
from exp_1209.models import TeacherStudentWithAdapter
from exp_1209.losses import CombinedLoss, compute_token_accuracy


def set_seed(seed: int = 42):
    """
    固定隨機種子以確保實驗可重現性

    Args:
        seed: 隨機種子值，預設為 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    loss_fn: CombinedLoss,
    device: str,
    epoch: int,
    distance_matrix: torch.Tensor,
    scaler=None,
    use_amp: bool = True,
) -> dict:
    """
    訓練一個 epoch

    Args:
        model: TeacherStudentWithAdapter 模型
        dataloader: 訓練資料載入器
        optimizer: 優化器
        loss_fn: CombinedLoss 損失函數
        device: 運算設備
        epoch: 當前 epoch 數
        distance_matrix: VQ 距離矩陣（監控用）
        scaler: GradScaler（AMP 用）
        use_amp: 是否使用混合精度

    Returns:
        包含各種 metrics 的 dict
    """
    model.train()
    model.adapter.train()  # 確保 adapter 在訓練模式

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0, 'ce_loss': 0,
        'token_acc': 0, 'distance_loss': 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        # Forward
        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)

            # 計算 loss
            loss, loss_info = loss_fn(
                student_out=output['student_adapted_out'],
                teacher_out=output['teacher_encoder_out'],
                codebook=output['codebook'],
                teacher_codes=output['teacher_codes'],
                original_encoder_out=output['student_encoder_out'],
            )

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
            optimizer.step()

        # 累計 metrics
        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['ce_loss'] += loss_info.get('ce_loss', 0)

        # Token accuracy
        token_acc = compute_token_accuracy(output['student_codes'], output['teacher_codes'])
        metrics['token_acc'] += token_acc

        # Distance loss（監控用）
        with torch.no_grad():
            s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
            t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
            s_flat = s_codes.reshape(-1).long()
            t_flat = t_codes.reshape(-1).long()
            dist = distance_matrix[s_flat, t_flat].mean().item()
            metrics['distance_loss'] += dist

        n_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{token_acc*100:.1f}%",
        })

    # 平均
    for k in metrics:
        metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    loss_fn: CombinedLoss,
    device: str,
    distance_matrix: torch.Tensor,
    use_amp: bool = True,
) -> dict:
    """
    驗證模型

    Args:
        model: 模型
        dataloader: 驗證資料載入器
        loss_fn: 損失函數
        device: 運算設備
        distance_matrix: VQ 距離矩陣
        use_amp: 是否使用混合精度

    Returns:
        包含各種 metrics 的 dict
    """
    model.eval()

    metrics = {
        'total_loss': 0, 'feature_loss': 0, 'triplet_loss': 0, 'ce_loss': 0,
        'token_acc': 0, 'distance_loss': 0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        with autocast(enabled=use_amp):
            output = model(noisy_audio, clean_audio)

            loss, loss_info = loss_fn(
                student_out=output['student_adapted_out'],
                teacher_out=output['teacher_encoder_out'],
                codebook=output['codebook'],
                teacher_codes=output['teacher_codes'],
                original_encoder_out=output['student_encoder_out'],
            )

        metrics['total_loss'] += loss_info.get('total_loss', loss.item())
        metrics['feature_loss'] += loss_info.get('feature_loss', 0)
        metrics['triplet_loss'] += loss_info.get('triplet_loss', 0)
        metrics['ce_loss'] += loss_info.get('ce_loss', 0)

        token_acc = compute_token_accuracy(output['student_codes'], output['teacher_codes'])
        metrics['token_acc'] += token_acc

        s_codes = output['student_codes'][0] if output['student_codes'].dim() == 3 else output['student_codes']
        t_codes = output['teacher_codes'][0] if output['teacher_codes'].dim() == 3 else output['teacher_codes']
        s_flat = s_codes.reshape(-1).long()
        t_flat = t_codes.reshape(-1).long()
        dist = distance_matrix[s_flat, t_flat].mean().item()
        metrics['distance_loss'] += dist

        n_batches += 1

    for k in metrics:
        metrics[k] /= n_batches

    return metrics


def plot_training_curves(history: dict, save_path: Path):
    """
    繪製訓練曲線

    Args:
        history: 訓練歷史記錄
        save_path: 儲存路徑
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs = range(1, len(history['train_total_loss']) + 1)

    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_total_loss'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    # Feature Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_feature_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_feature_loss'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Feature Loss')
    ax.set_title('Feature MSE Loss')
    ax.legend()
    ax.grid(True)

    # Triplet Loss
    ax = axes[0, 2]
    ax.plot(epochs, history['train_triplet_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_triplet_loss'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Triplet Loss')
    ax.set_title('Triplet Loss')
    ax.legend()
    ax.grid(True)

    # Token Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, [x*100 for x in history['train_token_acc']], 'b-', label='Train')
    ax.plot(epochs, [x*100 for x in history['val_token_acc']], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Token Accuracy (%)')
    ax.set_title('Token Accuracy')
    ax.legend()
    ax.grid(True)

    # Distance Loss
    ax = axes[1, 1]
    ax.plot(epochs, history['train_distance_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_distance_loss'], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance Loss')
    ax.set_title('VQ Distance (監控)')
    ax.legend()
    ax.grid(True)

    # CE Loss (if available)
    ax = axes[1, 2]
    if any(x > 0 for x in history.get('train_ce_loss', [0])):
        ax.plot(epochs, history['train_ce_loss'], 'b-', label='Train')
        ax.plot(epochs, history['val_ce_loss'], 'r-', label='Val')
        ax.set_title('CE Loss')
    else:
        ax.text(0.5, 0.5, 'CE Loss not used', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('CE Loss (未使用)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """
    主訓練函數
    """
    parser = argparse.ArgumentParser(description='Exp1209: Adapter + Triplet Loss')
    parser.add_argument('--exp_name', type=str, required=True, help='實驗名稱')
    parser.add_argument('--adapter_hidden', type=int, default=256, help='Adapter 隱藏層維度')
    parser.add_argument('--adapter_layers', type=int, default=2, help='Adapter 層數')
    parser.add_argument('--feature_weight', type=float, default=1.0, help='Feature Loss 權重')
    parser.add_argument('--triplet_weight', type=float, default=1.0, help='Triplet Loss 權重')
    parser.add_argument('--triplet_margin', type=float, default=0.5, help='Triplet margin')
    parser.add_argument('--ce_weight', type=float, default=0.0, help='CE Loss 權重')
    parser.add_argument('--ce_temperature', type=float, default=0.1, help='CE temperature')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='訓練 epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度')
    args = parser.parse_args()

    # 設定
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 輸出目錄
    exp_dir = Path(__file__).parent / 'experiments' / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 儲存配置
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exp1209: {args.exp_name}")
    print(f"{'='*60}")
    print(f"Adapter: hidden={args.adapter_hidden}, layers={args.adapter_layers}")
    print(f"Loss weights: feature={args.feature_weight}, triplet={args.triplet_weight}, ce={args.ce_weight}")
    print(f"{'='*60}\n")

    # 載入 distance matrix
    distance_matrix = torch.load(DISTANCE_MATRIX).to(device)

    # 創建模型
    print("Creating model...")
    model = TeacherStudentWithAdapter(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        adapter_hidden=args.adapter_hidden,
        adapter_layers=args.adapter_layers,
        device=device,
    )

    # 創建 loss function
    loss_fn = CombinedLoss(
        feature_weight=args.feature_weight,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        ce_weight=args.ce_weight,
        ce_temperature=args.ce_temperature,
    )

    # 只優化 adapter
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=0.01)

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # DataLoader
    class DataConfig:
        def __init__(self, batch_size, num_workers):
            self.use_hdf5 = False
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.pin_memory = True

    train_loader, val_loader = create_dataloaders(DataConfig(args.batch_size, args.num_workers))
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 訓練歷史
    history = {
        'train_total_loss': [], 'train_feature_loss': [], 'train_triplet_loss': [],
        'train_ce_loss': [], 'train_token_acc': [], 'train_distance_loss': [],
        'val_total_loss': [], 'val_feature_loss': [], 'val_triplet_loss': [],
        'val_ce_loss': [], 'val_token_acc': [], 'val_distance_loss': [],
    }

    best_val_acc = 0

    # 訓練迴圈
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # 訓練
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            distance_matrix, scaler, args.use_amp
        )

        # 驗證
        val_metrics = validate(model, val_loader, loss_fn, device, distance_matrix, args.use_amp)

        # 記錄
        for k in train_metrics:
            history[f'train_{k}'].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])

        # 打印
        print(f"\nTrain: loss={train_metrics['total_loss']:.4f}, "
              f"feat={train_metrics['feature_loss']:.4f}, "
              f"triplet={train_metrics['triplet_loss']:.4f}, "
              f"acc={train_metrics['token_acc']*100:.2f}%")
        print(f"Val:   loss={val_metrics['total_loss']:.4f}, "
              f"feat={val_metrics['feature_loss']:.4f}, "
              f"triplet={val_metrics['triplet_loss']:.4f}, "
              f"acc={val_metrics['token_acc']*100:.2f}%")

        # 儲存最佳模型
        if val_metrics['token_acc'] > best_val_acc:
            best_val_acc = val_metrics['token_acc']
            torch.save({
                'epoch': epoch,
                'adapter_state_dict': model.adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['token_acc'],
            }, exp_dir / 'best.pt')
            print(f"  ✓ New best: {best_val_acc*100:.2f}%")

        # 定期儲存
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'adapter_state_dict': model.adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, exp_dir / f'epoch_{epoch:03d}.pt')

            # 繪製曲線
            plot_training_curves(history, exp_dir / f'training_curves_epoch_{epoch:03d}.png')

    # 儲存最終結果
    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, exp_dir / 'training_curves_final.png')

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

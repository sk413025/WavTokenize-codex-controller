"""
Zero-Shot Denoising Transformer 訓練 (使用 VQ Distances)

新增功能:
1. 支持 SoftTargetLoss (Knowledge Distillation from VQ distances)
2. 支持 HybridDistanceLoss (Soft + Hard + Distribution Matching)
3. Loss Weight Warm-up (避免訓練初期不穩定)
4. 完整的分析和監控功能

使用範例:
    # Baseline (不使用 distances)
    python train_with_distances.py --exp_name baseline --loss_type ce
    
    # Soft Target (α=0.5)
    python train_with_distances.py --exp_name exp1_soft_05 --loss_type soft --alpha 0.5
    
    # Soft Target (α=0.7)
    python train_with_distances.py --exp_name exp2_soft_07 --loss_type soft --alpha 0.7
    
    # Hybrid Loss
    python train_with_distances.py --exp_name exp3_hybrid --loss_type hybrid \
        --alpha 0.3 --beta 0.3 --gamma 0.4
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import random
import json
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加必要的路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from decoder.pretrained import WavTokenizer
from model_zeroshot import ZeroShotDenoisingTransformer
from data_zeroshot_hdf5_v2 import (
    HDF5ZeroShotDataset,
    cached_collate_fn_with_distances
)
from losses_with_distances import SoftTargetLoss, HybridDistanceLoss
from config import PAD_TOKEN, CODEBOOK_SIZE


def set_seed(seed=42):
    """設置隨機種子以確保實驗可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file):
    """設置 logger"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_loss_weight_schedule(epoch, args):
    """
    動態調整 loss 權重（Warm-up 策略）
    
    前 warmup_epochs 只用 hard target (CE)
    之後線性增加 soft target 權重
    
    Returns:
        dict: {'alpha': float, 'beta': float, 'gamma': float}
    """
    if not args.use_warmup or epoch >= args.warmup_epochs:
        # 正常訓練階段：使用目標權重
        if args.loss_type == 'soft':
            return {'alpha': args.alpha, 'beta': 1.0 - args.alpha, 'gamma': 0.0}
        elif args.loss_type == 'hybrid':
            return {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma}
        else:  # ce
            return {'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0}
    else:
        # Warm-up 階段：只用 CE loss
        return {'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0}


def create_loss_function(args, epoch):
    """
    根據配置創建 loss function
    
    Args:
        args: 命令行參數
        epoch: 當前 epoch（用於 warm-up）
    
    Returns:
        criterion: Loss function
        use_distances: 是否需要 distances
    """
    weights = get_loss_weight_schedule(epoch, args)
    
    if args.loss_type == 'ce':
        # Baseline: 只用 Cross-Entropy
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        use_distances = False
        
    elif args.loss_type == 'soft':
        # Soft Target Loss
        if weights['alpha'] == 0.0:
            # Warm-up 階段：使用 CE
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            use_distances = False
        else:
            criterion = SoftTargetLoss(
                temperature=args.temperature,
                alpha=weights['alpha']
            )
            use_distances = True
            
    elif args.loss_type == 'hybrid':
        # Hybrid Loss
        if weights['alpha'] == 0.0:
            # Warm-up 階段：使用 CE
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            use_distances = False
        else:
            criterion = HybridDistanceLoss(
                alpha=weights['alpha'],
                beta=weights['beta'],
                gamma=weights['gamma'],
                temperature=args.temperature
            )
            use_distances = True
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")
    
    return criterion, use_distances


def analyze_token_predictions(pred_tokens, top_k=20):
    """
    分析 token 預測分布
    
    Returns:
        dict: 統計信息
    """
    all_tokens = pred_tokens.cpu().numpy().flatten()
    counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(counter)
    
    most_common = counter.most_common(top_k)
    
    # 計算熵（多樣性指標）
    probs = np.array([count / total_tokens for count in counter.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Top-1 token 佔比
    top1_ratio = most_common[0][1] / total_tokens if most_common else 0
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'unique_ratio': unique_tokens / CODEBOOK_SIZE,
        'entropy': entropy,
        'top1_token': most_common[0][0] if most_common else None,
        'top1_ratio': top1_ratio,
        'most_common': most_common[:10]
    }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, logger, args):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    # 如果使用 Hybrid Loss，記錄各個分量
    loss_components = {'soft': 0.0, 'hard': 0.0, 'wasserstein': 0.0}
    
    all_predictions = []
    
    # 確定是否需要 distances
    _, use_distances = create_loss_function(args, epoch)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        noisy_tokens = batch['noisy_tokens'].to(device)
        clean_tokens = batch['clean_tokens'].to(device)
        speaker_embeddings = batch['speaker_emb'].to(device)
        
        if use_distances:
            clean_distances = batch['clean_distances'].to(device)
        
        # Forward
        logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
        
        # 計算損失
        B, T, vocab = logits.shape
        
        if use_distances:
            # 使用 distances-based loss
            if isinstance(criterion, HybridDistanceLoss):
                # Hybrid Loss 返回字典
                loss_dict = criterion(logits, clean_distances, clean_tokens)
                loss = loss_dict['total_loss']
                
                # 記錄各個分量
                loss_components['soft'] += loss_dict['soft_loss']
                loss_components['hard'] += loss_dict['hard_loss']
                loss_components['wasserstein'] += loss_dict['wasserstein_loss']
            else:
                # Soft Target Loss
                loss = criterion(logits, clean_distances, clean_tokens)
        else:
            # Baseline CE Loss
            logits_flat = logits.reshape(B * T, vocab)
            clean_tokens_flat = clean_tokens.reshape(B * T)
            loss = criterion(logits_flat, clean_tokens_flat)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Token accuracy
        pred_tokens = logits.argmax(dim=-1)
        correct = (pred_tokens == clean_tokens).sum().item()
        total_correct += correct
        total_tokens += B * T
        
        # 收集預測用於分析
        all_predictions.append(pred_tokens.detach().flatten())
        
        # Update progress bar
        current_acc = (total_correct / total_tokens) * 100
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100
    
    # 分析 token 預測分布
    all_predictions = torch.cat(all_predictions, dim=0)
    token_stats = analyze_token_predictions(all_predictions)
    
    # 記錄到 log
    logger.info(f"  Token 預測分析:")
    logger.info(f"    - 唯一 token 數: {token_stats['unique_tokens']}/{CODEBOOK_SIZE} "
                f"({token_stats['unique_ratio']*100:.2f}%)")
    logger.info(f"    - 預測熵 (多樣性): {token_stats['entropy']:.4f}")
    logger.info(f"    - 最常見 token: {token_stats['top1_token']} "
                f"(佔比 {token_stats['top1_ratio']*100:.2f}%)")
    
    if token_stats['top1_ratio'] > 0.5:
        logger.warning(f"    ⚠️  警告: >50% 的預測都是同一個 token!")
    
    result = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'token_stats': token_stats
    }
    
    # 如果使用 Hybrid Loss，添加分量
    if isinstance(criterion, HybridDistanceLoss) and use_distances:
        result['loss_components'] = {
            'soft': loss_components['soft'] / len(dataloader),
            'hard': loss_components['hard'] / len(dataloader),
            'wasserstein': loss_components['wasserstein'] / len(dataloader)
        }
        logger.info(f"  Loss Components:")
        logger.info(f"    - Soft Loss: {result['loss_components']['soft']:.4f}")
        logger.info(f"    - Hard Loss: {result['loss_components']['hard']:.4f}")
        logger.info(f"    - Wasserstein Loss: {result['loss_components']['wasserstein']:.4f}")
    
    return result


def validate_epoch(model, dataloader, criterion, device, epoch, logger, args):
    """驗證一個 epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    loss_components = {'soft': 0.0, 'hard': 0.0, 'wasserstein': 0.0}
    all_predictions = []
    
    # 確定是否需要 distances
    _, use_distances = create_loss_function(args, epoch)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            noisy_tokens = batch['noisy_tokens'].to(device)
            clean_tokens = batch['clean_tokens'].to(device)
            speaker_embeddings = batch['speaker_emb'].to(device)
            
            if use_distances:
                clean_distances = batch['clean_distances'].to(device)
            
            # Forward
            logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
            
            # 計算損失
            B, T, vocab = logits.shape
            
            if use_distances:
                if isinstance(criterion, HybridDistanceLoss):
                    loss_dict = criterion(logits, clean_distances, clean_tokens)
                    loss = loss_dict['total_loss']
                    loss_components['soft'] += loss_dict['soft_loss']
                    loss_components['hard'] += loss_dict['hard_loss']
                    loss_components['wasserstein'] += loss_dict['wasserstein_loss']
                else:
                    loss = criterion(logits, clean_distances, clean_tokens)
            else:
                logits_flat = logits.reshape(B * T, vocab)
                clean_tokens_flat = clean_tokens.reshape(B * T)
                loss = criterion(logits_flat, clean_tokens_flat)
            
            total_loss += loss.item()
            
            # Token accuracy
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == clean_tokens).sum().item()
            total_correct += correct
            total_tokens += B * T
            
            all_predictions.append(pred_tokens.flatten())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100
    
    # 分析 token 預測分布
    all_predictions = torch.cat(all_predictions, dim=0)
    token_stats = analyze_token_predictions(all_predictions)
    
    logger.info(f"  Token 預測分析 (Validation):")
    logger.info(f"    - 唯一 token 數: {token_stats['unique_tokens']}/{CODEBOOK_SIZE} "
                f"({token_stats['unique_ratio']*100:.2f}%)")
    logger.info(f"    - 預測熵: {token_stats['entropy']:.4f}")
    logger.info(f"    - 最常見 token: {token_stats['top1_token']} "
                f"(佔比 {token_stats['top1_ratio']*100:.2f}%)")
    
    result = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'token_stats': token_stats
    }
    
    if isinstance(criterion, HybridDistanceLoss) and use_distances:
        result['loss_components'] = {
            'soft': loss_components['soft'] / len(dataloader),
            'hard': loss_components['hard'] / len(dataloader),
            'wasserstein': loss_components['wasserstein'] / len(dataloader)
        }
    
    return result


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, metrics_history, save_path):
    """保存 checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'metrics_history': metrics_history
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train with VQ Distances')
    
    # 實驗配置
    parser.add_argument('--exp_name', required=True, help='實驗名稱')
    parser.add_argument('--output_dir', default='./distance_experiments', help='輸出目錄')
    
    # 數據配置
    parser.add_argument('--cache_dir', default='./data_with_distances', help='Cached data 目錄')
    parser.add_argument('--batch_size', type=int, default=28, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    # Loss 配置
    parser.add_argument('--loss_type', choices=['ce', 'soft', 'hybrid'], required=True,
                       help='Loss type: ce (baseline), soft (soft target), hybrid (hybrid loss)')
    parser.add_argument('--temperature', type=float, default=2.0, help='Softmax temperature')
    parser.add_argument('--alpha', type=float, default=0.5, help='Soft target weight (for soft loss)')
    parser.add_argument('--beta', type=float, default=0.3, help='Hard target weight (for hybrid loss)')
    parser.add_argument('--gamma', type=float, default=0.4, help='Wasserstein weight (for hybrid loss)')
    
    # Warm-up 配置
    parser.add_argument('--use_warmup', action='store_true', help='使用 loss weight warm-up')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='Warm-up epochs')
    
    # 模型配置
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 維度')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN 維度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--fusion_method', default='add', choices=['add', 'concat', 'film', 'cross_attn'],
                       help='Speaker fusion method')
    
    # 訓練配置
    parser.add_argument('--num_epochs', type=int, default=200, help='訓練 epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='梯度裁剪 (0=不裁剪)')
    
    # Scheduler 配置
    parser.add_argument('--use_scheduler', action='store_true', help='使用 LR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Scheduler patience')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='LR decay factor')
    
    # Early Stopping 配置
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early stopping patience')
    
    # Checkpoint 配置
    parser.add_argument('--save_checkpoint_freq', type=int, default=50, help='保存 checkpoint 頻率')
    
    # WavTokenizer 配置
    parser.add_argument('--wavtokenizer_config',
                       default='../../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                       help='WavTokenizer 配置')
    parser.add_argument('--wavtokenizer_checkpoint',
                       default='../../models/wavtokenizer_large_speech_320_24k.ckpt',
                       help='WavTokenizer checkpoint')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 創建輸出目錄
    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # 設置 logger
    logger = setup_logger(log_dir / 'training.log')
    
    # 保存配置
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"實驗名稱: {args.exp_name}")
    logger.info(f"Loss Type: {args.loss_type}")
    logger.info(f"輸出目錄: {exp_dir}")
    logger.info("=" * 80)
    
    # 載入數據
    logger.info("載入數據...")
    train_dataset = HDF5ZeroShotDataset(
        f'{args.cache_dir}/cache_with_distances.h5',
        split='train'
    )
    val_dataset = HDF5ZeroShotDataset(
        f'{args.cache_dir}/cache_with_distances.h5',
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cached_collate_fn_with_distances,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cached_collate_fn_with_distances,
        pin_memory=True
    )
    
    logger.info(f"✓ 訓練集: {len(train_dataset)} 樣本")
    logger.info(f"✓ 驗證集: {len(val_dataset)} 樣本")
    
    # 載入 WavTokenizer (獲取 codebook)
    logger.info("載入 WavTokenizer...")
    device = torch.device(args.device)
    wavtokenizer = WavTokenizer.from_pretrained0802(
        args.wavtokenizer_config,
        args.wavtokenizer_checkpoint
    )
    wavtokenizer = wavtokenizer.to(device)
    
    # 提取 codebook
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed  # (4096, 512)
    logger.info(f"✓ Codebook shape: {codebook.shape}")
    
    # 創建模型
    logger.info("創建模型...")
    model = ZeroShotDenoisingTransformer(
        codebook=codebook,
        speaker_embed_dim=192,  # ⭐ 使用 192 而非 256
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        fusion_method=args.fusion_method
    ).to(device)
    
    # 統計參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ 總參數量: {total_params:,}")
    logger.info(f"✓ 可訓練參數量: {trainable_params:,}")
    
    # 創建優化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 創建 scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=1e-6
        )
        logger.info(f"✓ 使用 ReduceLROnPlateau scheduler (patience={args.scheduler_patience})")
    
    # 訓練循環
    logger.info("開始訓練...")
    best_val_acc = 0.0
    patience_counter = 0
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"{'='*80}")
        
        # 創建本 epoch 的 loss function
        criterion, use_distances = create_loss_function(args, epoch)
        
        # 如果在 warm-up 階段，記錄
        if args.use_warmup and epoch <= args.warmup_epochs:
            logger.info(f"⏳ Warm-up 階段 ({epoch}/{args.warmup_epochs}): 只使用 CE Loss")
        
        # 訓練
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger, args)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        
        # 驗證
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, logger, args)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # 記錄 metrics
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['train_acc'].append(train_metrics['accuracy'])
        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['val_acc'].append(val_metrics['accuracy'])
        metrics_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 更新 scheduler
        if scheduler:
            scheduler.step(val_metrics['accuracy'])
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # 保存 best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_acc, metrics_history,
                checkpoint_dir / 'best_model.pt'
            )
            logger.info(f"✅ 保存 Best Model (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # 定期保存 checkpoint
        if epoch % args.save_checkpoint_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_acc, metrics_history,
                checkpoint_dir / f'epoch_{epoch:03d}.pt'
            )
            logger.info(f"💾 保存 Checkpoint (Epoch {epoch})")
        
        # Early Stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"⏹️  Early Stopping at Epoch {epoch} (no improvement for {args.early_stopping_patience} epochs)")
            break
    
    # 訓練結束
    logger.info(f"\n{'='*80}")
    logger.info(f"訓練完成！")
    logger.info(f"Best Val Accuracy: {best_val_acc:.2f}%")
    logger.info(f"{'='*80}")
    
    # 保存 metrics history
    with open(log_dir / 'metrics_history.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    logger.info(f"✓ Metrics history 已保存到 {log_dir / 'metrics_history.json'}")


if __name__ == '__main__':
    main()

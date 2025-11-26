"""
Exp_1126 訓練腳本：使用 1D Wasserstein Distance Loss with Loss Scaling

基於 commit 0502ca619f86f1d336eba0eb23e507b46207eca5 (exp5-1-2) 重現實驗
目標: 探索 Loss Scaling 對 Wasserstein Loss 訓練效果的影響

實驗背景:
- Wasserstein Loss ≈ 0.38 (僅為 CE 的 4.36%)
- Wasserstein gradient ≈ CE gradient 的 0.59%
- 使用 Loss Scaling Factor: 23.0x 使 Wasserstein Loss 與 CE Loss 同量級
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime

# 添加路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # c_code/
sys.path.insert(0, str(Path(__file__).parent))  # 當前目錄
sys.path.insert(0, str(Path(__file__).parent / 'shared'))  # shared 目錄

# 導入模組
from decoder.pretrained import WavTokenizer
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn
from config import PAD_TOKEN
from model_zeroshot_lora import ZeroShotDenoisingTransformer
from shared.visualization_utils import save_loss_plot
from wasserstein_loss_1d import Wasserstein1DLoss, HybridWasserstein1DCELoss


def save_audio_samples_exp_1126(model, wavtokenizer, codebook, train_loader, val_loader, device, output_dir, epoch, num_samples=3):
    """
    簡化版音頻樣本保存（專為 Exp_1126 設計）

    Args:
        model: 訓練中的模型
        wavtokenizer: WavTokenizer 模型
        codebook: WavTokenizer codebook
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        device: 計算設備
        output_dir: 輸出目錄
        epoch: 當前 epoch
        num_samples: 每個數據集保存的樣本數
    """
    from shared.visualization_utils import save_spectrogram_comparison
    import soundfile as sf
    
    model.eval()
    
    for split, loader in [("training", train_loader), ("validation", val_loader)]:
        sample_dir = output_dir / "audio_samples" / f"epoch_{epoch}_{split}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if saved_count >= num_samples:
                    break
                    
                noisy_tokens = batch['noisy_tokens'][:1].to(device)
                clean_tokens = batch['clean_tokens'][:1].to(device)
                speaker_embeddings = batch['speaker_embeddings'][:1].to(device)
                
                # 模型預測
                pred_logits = model(noisy_tokens, speaker_embeddings, return_logits=True)
                pred_tokens = pred_logits.argmax(dim=-1)

                # Clamp tokens to valid WavTokenizer range [0, 4095]
                # 模型可能預測 PAD_TOKEN (4096) 或其他無效值
                noisy_tokens_clamped = torch.clamp(noisy_tokens.squeeze(0), 0, 4095)
                pred_tokens_clamped = torch.clamp(pred_tokens.squeeze(0), 0, 4095)
                clean_tokens_clamped = torch.clamp(clean_tokens.squeeze(0), 0, 4095)

                # 使用 codebook 將 tokens 轉為 features
                noisy_features = codebook[noisy_tokens_clamped].permute(1, 0).unsqueeze(0)
                pred_features = codebook[pred_tokens_clamped].permute(1, 0).unsqueeze(0)
                clean_features = codebook[clean_tokens_clamped].permute(1, 0).unsqueeze(0)
                
                # 使用 WavTokenizer 解碼
                bandwidth_id = torch.tensor([0]).to(device)
                noisy_audio = wavtokenizer.decode(noisy_features, bandwidth_id=bandwidth_id).cpu().squeeze().numpy()
                pred_audio = wavtokenizer.decode(pred_features, bandwidth_id=bandwidth_id).cpu().squeeze().numpy()
                clean_audio = wavtokenizer.decode(clean_features, bandwidth_id=bandwidth_id).cpu().squeeze().numpy()
                
                # 保存音頻
                idx = saved_count
                sf.write(sample_dir / f"sample_{idx}_noisy.wav", noisy_audio, 24000)
                sf.write(sample_dir / f"sample_{idx}_pred.wav", pred_audio, 24000)
                sf.write(sample_dir / f"sample_{idx}_clean.wav", clean_audio, 24000)
                
                # 保存頻譜圖
                save_spectrogram_comparison(
                    torch.from_numpy(noisy_audio),
                    torch.from_numpy(clean_audio),
                    torch.from_numpy(pred_audio),
                    sample_dir / f"sample_{idx}_spectrogram.png",
                    sample_rate=24000,
                    title=f"{split.upper()} Sample {idx} - Epoch {epoch}"
                )
                
                saved_count += 1
                
        logger.info(f"保存了 {saved_count} 個 {split} 音頻樣本（epoch {epoch}）")
    
    model.train()


def set_seed(seed=42):
    """
    設置隨機種子以確保實驗可重複性

    Args:
        seed: 隨機種子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(output_dir):
    """
    設置 logger 用於記錄訓練過程

    Args:
        output_dir: 輸出目錄路徑

    Returns:
        logger 對象
    """
    log_file = output_dir / 'training.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, criterion, device, logger, epoch, args):
    """
    訓練一個 epoch

    Args:
        model: 訓練中的模型
        train_loader: 訓練數據加載器
        optimizer: 優化器
        criterion: 損失函數
        device: 計算設備
        logger: 日誌記錄器
        epoch: 當前 epoch 編號
        args: 命令行參數

    Returns:
        avg_loss: 平均損失
        accuracy: 訓練準確率
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        noisy_tokens = batch['noisy_tokens'].to(device)
        clean_tokens = batch['clean_tokens'].to(device)
        speaker_embeddings = batch['speaker_embeddings'].to(device)

        # Forward
        logits = model(noisy_tokens, speaker_embeddings, return_logits=True)

        # 對齊長度
        T = min(noisy_tokens.shape[1], clean_tokens.shape[1])
        logits = logits[:, :T, :].contiguous()
        target = clean_tokens[:, :T].contiguous()

        # 計算 loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 統計
        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        correct += (predictions == target).sum().item()
        total += (target != PAD_TOKEN).sum().item()

        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total if total > 0 else 0

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, logger):
    """
    驗證模型

    Args:
        model: 訓練中的模型
        val_loader: 驗證數據加載器
        criterion: 損失函數
        device: 計算設備
        logger: 日誌記錄器

    Returns:
        avg_loss: 平均驗證損失
        accuracy: 驗證準確率
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            noisy_tokens = batch['noisy_tokens'].to(device)
            clean_tokens = batch['clean_tokens'].to(device)
            speaker_embeddings = batch['speaker_embeddings'].to(device)

            logits = model(noisy_tokens, speaker_embeddings, return_logits=True)

            T = min(noisy_tokens.shape[1], clean_tokens.shape[1])
            logits = logits[:, :T, :].contiguous()
            target = clean_tokens[:, :T].contiguous()

            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            total_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            correct += (predictions == target).sum().item()
            total += (target != PAD_TOKEN).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total if total > 0 else 0

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, output_dir, args, is_best=False):
    """
    保存模型 checkpoint

    Args:
        model: 訓練中的模型
        optimizer: 優化器
        scheduler: 學習率調度器
        epoch: 當前 epoch
        val_acc: 驗證準確率
        val_loss: 驗證損失
        output_dir: 輸出目錄
        args: 命令行參數
        is_best: 是否為最佳模型
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'args': vars(args)
    }

    if is_best:
        save_path = output_dir / 'best_model.pt'
    else:
        save_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'

    torch.save(checkpoint, save_path)
    logger.info(f"保存 checkpoint: {save_path}")


def main():
    """主函式"""
    parser = argparse.ArgumentParser(description='Exp_1126: 1D Wasserstein Loss with Scaling')

    # Data paths
    parser.add_argument('--train_cache', type=str, required=True, help='訓練緩存路徑')
    parser.add_argument('--val_cache', type=str, required=True, help='驗證緩存路徑')
    parser.add_argument('--wavtok_config', type=str, required=True, help='WavTokenizer 配置文件路徑')
    parser.add_argument('--wavtok_ckpt', type=str, required=True, help='WavTokenizer checkpoint 路徑')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=28, help='批次大小')
    parser.add_argument('--epochs', type=int, default=120, help='訓練 epoch 數')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='權重衰減')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout 率')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # Wasserstein Loss hyperparameters
    parser.add_argument('--wasserstein_alpha', type=float, default=1.0, help='Wasserstein 權重')
    parser.add_argument('--wasserstein_scale_factor', type=float, default=23.0, help='Loss 縮放因子')
    parser.add_argument('--use_1d_wasserstein', action='store_true', default=True, help='使用 1D Wasserstein')

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 維度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力頭數')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN 維度')
    parser.add_argument('--fusion_method', type=str, default='cross_attn', help='融合方法')
    parser.add_argument('--use_learnable_gate', action='store_true', default=True, help='使用可學習門控')

    # Other settings
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'none'])
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_interval', type=int, default=20, help='保存間隔')
    parser.add_argument('--num_vis_samples', type=int, default=3, help='可視化樣本數')
    parser.add_argument('--device', type=str, default='cuda', help='計算設備')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--resume', type=str, default=None, help='恢復訓練的 checkpoint 路徑')
    parser.add_argument('--disable_visualization', action='store_true', help='禁用可視化')
    parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')

    args = parser.parse_args()

    # 設置輸出目錄
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / 'runs' / f'exp_1126_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 設置 logger
    global logger
    logger = setup_logger(output_dir)

    logger.info("=" * 80)
    logger.info("Exp_1126: 1D Wasserstein Loss with Loss Scaling")
    logger.info("基於 commit 0502ca619f86f1d336eba0eb23e507b46207eca5 重現")
    logger.info("=" * 80)

    # 保存配置
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"保存配置至 {config_path}")

    # 設置隨機種子
    set_seed(args.seed)
    logger.info(f"隨機種子: {args.seed}")

    # 載入 WavTokenizer
    logger.info("載入 WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(args.wavtok_config, args.wavtok_ckpt)
    wavtokenizer = wavtokenizer.to(args.device)
    wavtokenizer.eval()
    for param in wavtokenizer.parameters():
        param.requires_grad = False
    logger.info("WavTokenizer 載入完成並凍結!")

    # 獲取 codebook
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    logger.info(f"Codebook 形狀: {codebook.shape}")

    # 創建數據集
    logger.info("載入數據集...")
    train_dataset = ZeroShotAudioDatasetCached(args.train_cache)
    val_dataset = ZeroShotAudioDatasetCached(args.val_cache)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cached_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cached_collate_fn,
        pin_memory=True
    )

    logger.info(f"訓練樣本數: {len(train_dataset)}")
    logger.info(f"驗證樣本數: {len(val_dataset)}")

    # 創建模型
    logger.info("創建模型 (Cross-Attention with Learnable Gate)...")
    model = ZeroShotDenoisingTransformer(
        codebook=codebook,
        speaker_embed_dim=256,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        fusion_method=args.fusion_method,
        use_learnable_gate=args.use_learnable_gate,
    )
    model = model.to(args.device)
    logger.info("模型創建完成!")

    # 打印參數統計
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"總參數: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"可訓練參數: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    logger.info(f"凍結參數: {total_params - trainable_params:,}")

    # 創建優化器
    logger.info("創建優化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    logger.info(f"優化器: AdamW (lr={args.learning_rate}, wd={args.weight_decay})")

    # 創建學習率調度器
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.patience, factor=0.5, verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # Loss - 1D Wasserstein with Scaling
    logger.info(f"使用 Hybrid 1D Wasserstein-CE Loss (alpha={args.wasserstein_alpha})")
    logger.info("✓ 1D Wasserstein: 內存友好, O(n) 複雜度")
    logger.info(f"✓ Wasserstein Loss Scaling: {args.wasserstein_scale_factor:.2f}x")
    criterion = HybridWasserstein1DCELoss(
        num_classes=4096,
        alpha=args.wasserstein_alpha,
        scale_factor=args.wasserstein_scale_factor
    ).to(args.device)

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        logger.info(f"從 checkpoint 恢復: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        logger.info(f"從 epoch {start_epoch} 恢復, 最佳 val acc: {best_val_acc:.2f}%")

    # 訓練循環
    logger.info("開始訓練...")
    logger.info("=" * 80)

    # 追蹤歷史用於繪圖
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 80)

        # 訓練
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, args.device, logger, epoch+1, args
        )
        logger.info(f"訓練 Loss: {train_loss:.4f}, 訓練 Acc: {train_acc:.2f}%")

        # 驗證
        val_loss, val_acc = validate(model, val_loader, criterion, args.device, logger)
        logger.info(f"驗證 Loss: {val_loss:.4f}, 驗證 Acc: {val_acc:.2f}%")

        # 記錄歷史
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # 學習率調度
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 更新最佳指標
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, val_loss,
                output_dir, args, is_best=True
            )

        # 定期保存 checkpoint 和可視化
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, val_loss,
                output_dir, args, is_best=False
            )

            if not args.disable_visualization:
                logger.info(f"保存可視化內容 (epoch {epoch+1})...")
                try:
                    save_loss_plot(
                        train_losses=train_loss_history,
                        val_losses=val_loss_history,
                        train_accs=train_acc_history,
                        val_accs=val_acc_history,
                        output_path=output_dir / f"loss_curves_epoch_{epoch+1}.png",
                        title=f"Exp_1126 Training Progress - Epoch {epoch+1}"
                    )

                    save_audio_samples_exp_1126(
                        model=model,
                        wavtokenizer=wavtokenizer,
                        codebook=codebook,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=args.device,
                        output_dir=output_dir,
                        epoch=epoch+1,
                        num_samples=args.num_vis_samples
                    )
                except Exception as e:
                    logger.warning(f"保存可視化內容失敗: {str(e)}")

        logger.info(f"目前最佳 Val Acc: {best_val_acc:.2f}%")
        logger.info("=" * 80)

    # 保存最終結果
    logger.info("保存最終 loss 曲線...")
    try:
        save_loss_plot(
            train_losses=train_loss_history,
            val_losses=val_loss_history,
            train_accs=train_acc_history,
            val_accs=val_acc_history,
            output_path=output_dir / "final_loss_curves.png",
            title=f"Exp_1126 - Final Training Results (Best Val Acc: {best_val_acc:.2f}%)"
        )
    except Exception as e:
        logger.warning(f"保存最終 loss 曲線失敗: {str(e)}")

    logger.info("訓練完成!")
    logger.info(f"最佳 Val Acc: {best_val_acc:.2f}%")
    logger.info(f"結果保存至: {output_dir}")


if __name__ == '__main__':
    main()

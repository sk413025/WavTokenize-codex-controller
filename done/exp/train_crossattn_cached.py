"""
Zero-Shot Speaker Denoising Transformer - Cross-Attention Fusion 版本

實驗編號: EXP-20251105-CrossAttn
目的: 驗證假設 2 - Speaker Embedding 影響力不足
改進: 將 Additive Fusion 改為 Cross-Attention Mechanism

關鍵改變:
1. ✅ 使用 ZeroShotDenoisingTransformerCrossAttn (Cross-Attention Fusion)
2. ✅ Batch size: 64 (大幅提升，從 28)
3. ✅ 無 weight decay (weight_decay=0.0)
4. ✅ 無 scheduler (移除 ReduceLROnPlateau)
5. ✅ 固定學習率: 1e-4

使用方式:
    python train_crossattn_cached.py \\
        --cache_dir ./data \\
        --output_dir ./results/crossattn_100epochs \\
        --num_epochs 100 \\
        --batch_size 64
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# 添加必要的路徑
sys.path.insert(0, str(Path(__file__).parent.parent))  # 加入 done/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # 加入 c_code/

from decoder.pretrained import WavTokenizer
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 導入 Cross-Attention 模組
from model_zeroshot_crossattn import ZeroShotDenoisingTransformerCrossAttn
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def plot_spectrograms(noisy_audio, pred_audio, clean_audio, save_path):
    """繪製三個音頻的頻譜圖"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, (audio, title) in enumerate([
        (noisy_audio, 'Noisy Audio'),
        (pred_audio, 'Predicted Audio'),
        (clean_audio, 'Clean Audio (Target)')
    ]):
        # 計算 STFT
        D = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # 繪製
        img = librosa.display.specshow(
            D_db,
            y_axis='log',
            x_axis='time',
            sr=24000,
            hop_length=512,
            ax=axes[idx],
            cmap='viridis'
        )
        axes[idx].set_title(title, fontsize=14)
        axes[idx].set_ylabel('Frequency (Hz)')
        fig.colorbar(img, ax=axes[idx], format='%+2.0f dB')

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    """繪製訓練和驗證的損失及準確率曲線"""
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. CE Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CE Loss')
    axes[0].set_title('CrossEntropy Loss (Cross-Attention)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Token Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Token Accuracy (Cross-Attention)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {output_path}")


def setup_logger(output_dir):
    """設置 logger"""
    log_file = Path(output_dir) / 'training.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    訓練一個 epoch

    Args:
        model: ZeroShotDenoisingTransformerCrossAttn
        dataloader: DataLoader
        optimizer: optimizer
        criterion: loss function
        device: torch device
        epoch: current epoch number
    """
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    # 禁用 tqdm 輸出（避免日誌過大）
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=True)

    for batch_idx, batch in enumerate(progress_bar):
        # 從緩存接收已處理好的數據
        noisy_tokens = batch['noisy_tokens'].to(device)
        clean_tokens = batch['clean_tokens'].to(device)
        speaker_embeddings = batch['speaker_embeddings'].to(device)

        optimizer.zero_grad()

        # 模型前向傳播
        logits = model(noisy_tokens, speaker_embeddings, return_logits=True)  # (B, T, 4096)

        # 計算損失
        B, T, vocab = logits.shape
        logits_flat = logits.reshape(B * T, vocab)
        clean_tokens_flat = clean_tokens.reshape(B * T)
        loss = criterion(logits_flat, clean_tokens_flat)

        # 反向傳播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 統計
        total_loss += loss.item()

        # Token accuracy
        pred_tokens = logits.argmax(dim=-1)
        correct = (pred_tokens == clean_tokens).sum().item()
        total_correct += correct
        total_tokens += B * T

        # 更新進度條
        current_acc = (total_correct / total_tokens) * 100
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def validate_epoch(model, dataloader, criterion, device, epoch):
    """驗證一個 epoch"""
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        # 禁用 tqdm 輸出（避免日誌過大）
        for batch in tqdm(dataloader, desc="Validation", disable=True):
            noisy_tokens = batch['noisy_tokens'].to(device)
            clean_tokens = batch['clean_tokens'].to(device)
            speaker_embeddings = batch['speaker_embeddings'].to(device)

            # Forward
            logits = model(noisy_tokens, speaker_embeddings, return_logits=True)

            # 計算損失
            B, T, vocab = logits.shape
            logits_flat = logits.reshape(B * T, vocab)
            clean_tokens_flat = clean_tokens.reshape(B * T)
            loss = criterion(logits_flat, clean_tokens_flat)

            total_loss += loss.item()

            # Token accuracy
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == clean_tokens).sum().item()
            total_correct += correct
            total_tokens += B * T

    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='Cross-Attention Fusion - Zero-Shot Denoising')

    # 緩存參數
    parser.add_argument('--cache_dir', default='./data', help='緩存目錄')
    parser.add_argument('--output_dir', default=None, help='輸出目錄 (默認自動生成)')

    # 模型參數
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 維度')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN 維度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--speaker_tokens', type=int, default=4, help='Speaker tokens K (>1 可避免注意力退化)')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='訓練 epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率 (固定)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # WavTokenizer 參數
    parser.add_argument('--wavtokenizer_config',
                       default='../../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                       help='WavTokenizer 配置文件')
    parser.add_argument('--wavtokenizer_checkpoint',
                       default='../../models/wavtokenizer_large_speech_320_24k.ckpt',
                       help='WavTokenizer checkpoint')

    args = parser.parse_args()

    # 自動生成輸出目錄 (帶時間戳)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f'./results/crossattn_100epochs_{timestamp}'

    # 檢查緩存是否存在
    cache_dir = Path(args.cache_dir)
    train_cache_path = cache_dir / 'train_cache.pt'
    val_cache_path = cache_dir / 'val_cache.pt'
    config_cache_path = cache_dir / 'cache_config.pt'

    if not train_cache_path.exists():
        raise FileNotFoundError(
            f"訓練集緩存不存在: {train_cache_path}\n"
            f"請先運行 preprocess_zeroshot_cache.py 生成緩存"
        )
    if not val_cache_path.exists():
        raise FileNotFoundError(
            f"驗證集緩存不存在: {val_cache_path}\n"
            f"請先運行 preprocess_zeroshot_cache.py 生成緩存"
        )

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'audio_samples').mkdir(parents=True, exist_ok=True)

    # 設置 logger
    logger = setup_logger(output_dir)
    logger.info("=" * 80)
    logger.info("Cross-Attention Speaker Fusion Experiment")
    logger.info("=" * 80)
    logger.info(f"實驗編號: EXP-20251105-CrossAttn")
    logger.info(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # 讀取緩存配置
    if config_cache_path.exists():
        cache_config = torch.load(config_cache_path)
        logger.info("緩存配置:")
        logger.info(f"  - Speaker Encoder: {cache_config.get('speaker_encoder', 'unknown')}")
        logger.info(f"  - Speaker Dim: {cache_config.get('speaker_dim', 'unknown')}")
        logger.info(f"  - Train Samples: {cache_config.get('train_samples', 'unknown')}")
        logger.info(f"  - Val Samples: {cache_config.get('val_samples', 'unknown')}")
        speaker_dim = cache_config.get('speaker_dim', 256)
    else:
        logger.warning("未找到緩存配置文件，使用默認值")
        speaker_dim = 256

    # 保存配置
    config_path = output_dir / 'config.json'
    config_dict = vars(args).copy()
    config_dict.update({
        'experiment_id': 'EXP-20251105-CrossAttn',
        'fusion_type': 'cross_attention',
        'weight_decay': 0.0,
        'scheduler': None,
        'hypothesis': 'Speaker Embedding 影響力不足',
        'improvement': 'Cross-Attention 替代 Additive Fusion'
    })
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"配置已保存至: {config_path}")

    # 設備 - 強制使用 GPU 2
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device} (物理 GPU 2)")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # 載入 WavTokenizer (僅用於生成音頻樣本)
    logger.info("載入 WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(
        args.wavtokenizer_config,
        args.wavtokenizer_checkpoint
    )
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()

    # 提取 Codebook
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    logger.info(f"Codebook 形狀: {codebook.shape}")  # (4096, 512)

    # 載入緩存數據集
    logger.info("=" * 80)
    logger.info("載入緩存數據集...")
    logger.info("=" * 80)

    train_dataset = ZeroShotAudioDatasetCached(str(train_cache_path))
    val_dataset = ZeroShotAudioDatasetCached(str(val_cache_path))

    logger.info(f"✓ 訓練集: {len(train_dataset)} 樣本")
    logger.info(f"✓ 驗證集: {len(val_dataset)} 樣本")

    # DataLoader
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

    logger.info(f"✓ DataLoader 配置:")
    logger.info(f"  - Batch size: {args.batch_size} (大幅提升)")
    logger.info(f"  - Num workers: {args.num_workers}")
    logger.info(f"  - Pin memory: True")

    # 創建 Cross-Attention 模型
    logger.info("=" * 80)
    logger.info("創建 Cross-Attention Denoising Transformer")
    logger.info("=" * 80)
    model = ZeroShotDenoisingTransformerCrossAttn(
        codebook=codebook,
        speaker_embed_dim=speaker_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        speaker_tokens=args.speaker_tokens
    ).to(device)

    # 計算參數量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_params = sum(b.numel() for b in model.buffers())
    total_params = trainable_params + buffer_params

    # Cross-Attention 新增參數
    cross_attn_params = sum(p.numel() for n, p in model.named_parameters() 
                           if 'cross_attn_fusion' in n)

    logger.info("模型架構:")
    logger.info(f"  - d_model: {args.d_model}")
    logger.info(f"  - nhead: {args.nhead}")
    logger.info(f"  - num_layers: {args.num_layers}")
    logger.info(f"  - speaker_dim: {speaker_dim}")
    logger.info(f"  - fusion_type: Cross-Attention (NEW)")
    logger.info(f"  - speaker_tokens (K): {args.speaker_tokens}")
    logger.info("")
    logger.info(f"模型總參數數量: {total_params:,}")
    logger.info(f"  - 可訓練參數: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"  - Codebook (buffer): {buffer_params:,} ({buffer_params/total_params*100:.2f}%)")
    logger.info(f"  - Cross-Attention 新增: {cross_attn_params:,} (~{cross_attn_params/1e6:.2f}M)")
    logger.info("  - ECAPA-TDNN: 已預計算，不在模型內")
    logger.info("=" * 80)

    # 損失函數
    criterion = nn.CrossEntropyLoss()

    # 優化器 (無 weight decay)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    logger.info("訓練配置:")
    logger.info(f"  - Optimizer: Adam")
    logger.info(f"  - Learning Rate: {args.learning_rate} (固定)")
    logger.info(f"  - Weight Decay: 0.0 (無)")
    logger.info(f"  - Scheduler: None (無)")
    logger.info(f"  - Gradient Clipping: 1.0")
    logger.info("")

    # 訓練歷史記錄
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # 訓練循環
    logger.info("開始訓練...")
    logger.info("")

    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, args.num_epochs + 1):
        # 訓練
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.2f}%")

        # 驗證
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.2f}%")

        # 記錄訓練歷史
        train_loss_history.append(train_metrics['loss'])
        val_loss_history.append(val_metrics['loss'])
        train_acc_history.append(train_metrics['accuracy'])
        val_acc_history.append(val_metrics['accuracy'])

        # 當前學習率 (固定)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning Rate: {current_lr:.2e} (固定)")

        # 保存 checkpoint (每 5 epochs)
        if epoch % 5 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history
            }, checkpoint_path)
            logger.info(f"  ✓ 保存 checkpoint: checkpoint_epoch_{epoch}.pth")

        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'val_metrics': val_metrics
            }, best_model_path)
            logger.info(f"  ✓ 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")

        # 每 10 epochs 繪製損失曲線
        if epoch % 10 == 0 and epoch > 0:
            logger.info(f"  繪製訓練曲線...")
            try:
                plot_path = output_dir / f'loss_curves_epoch_{epoch}.png'
                plot_loss_curves(
                    train_loss_history,
                    val_loss_history,
                    train_acc_history,
                    val_acc_history,
                    str(plot_path)
                )
                logger.info(f"  ✓ 已保存損失曲線到: loss_curves_epoch_{epoch}.png")
            except Exception as e:
                logger.error(f"  繪製損失曲線時出錯: {e}")

        logger.info("")

    # 最終總結
    logger.info("=" * 80)
    logger.info("Cross-Attention 實驗完成！")
    logger.info("=" * 80)
    logger.info(f"最佳驗證準確率: {best_val_acc:.2f}%")
    logger.info(f"最佳驗證損失: {best_val_loss:.4f}")
    logger.info("")

    # 與 Baseline 對比
    baseline_acc = 38.57  # 原始 Additive Fusion 的 Val Acc (Epoch 77)
    logger.info("結果判斷:")

    if best_val_acc > baseline_acc + 5.0:
        logger.info(f"  ✅ Val Acc {best_val_acc:.2f}% > Baseline {baseline_acc:.2f}% + 5%")
        logger.info(f"  ✅ 相比 Baseline，提升了 {best_val_acc - baseline_acc:.2f}%")
        logger.info("  ✅ 假設 2 驗證成功！Cross-Attention 顯著改善 Speaker Influence")
    elif best_val_acc > baseline_acc:
        logger.info(f"  ⚠️  Val Acc {best_val_acc:.2f}% > Baseline {baseline_acc:.2f}%")
        logger.info(f"  ⚠️  相比 Baseline，提升了 {best_val_acc - baseline_acc:.2f}%")
        logger.info("  ⚠️  略有改善，但未達顯著水平")
    else:
        logger.info(f"  ❌ Val Acc {best_val_acc:.2f}% ≤ Baseline {baseline_acc:.2f}%")
        logger.info(f"  ❌ 未能超越 Baseline")
        logger.info("  ❌ 假設 2 可能不成立，問題可能在其他地方")

    # 繪製最終損失曲線
    logger.info("")
    logger.info("=" * 80)
    logger.info("生成可視化結果")
    logger.info("=" * 80)

    plot_path = output_dir / 'loss_curves_final.png'
    plot_loss_curves(
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        str(plot_path)
    )
    logger.info(f"✓ 損失曲線已保存: {plot_path.name}")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 所有結果已保存至: {output_dir}")
    logger.info("=" * 80)
    logger.info("生成的文件:")
    logger.info(f"  - training.log: 訓練日誌")
    logger.info(f"  - config.json: 實驗配置")
    logger.info(f"  - loss_curves_final.png: 最終損失曲線")
    logger.info(f"  - best_model.pth: 最佳模型")
    logger.info(f"  - checkpoint_epoch_*.pth: 每 10 epochs 的 checkpoint")


if __name__ == '__main__':
    main()

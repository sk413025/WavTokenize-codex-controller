"""
Token Denoising Transformer 訓練腳本 (Baseline + Speaker Embedding L2 Loss)

目標: 通過引入 Speaker Embedding L2 Loss 作為輔助約束，提升模型對未見語者的泛化能力

使用方式:
    python train_with_speaker.py \\
        --input_dirs /path/to/noisy1 /path/to/noisy2 \\
        --target_dir /path/to/clean \\
        --output_dir ./results/exp2/speaker_loss_lambda0.5 \\
        --lambda_speaker 0.5 \\
        --num_epochs 600 \\
        --batch_size 8
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

# 添加必要的路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'exp'))

from data import AudioDataset
from model import TokenDenoisingTransformer
from loss_with_speaker import create_loss_with_speaker

from decoder.pretrained import WavTokenizer
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def save_audio_samples(
    wavtokenizer,
    noisy_tokens,
    pred_tokens,
    clean_tokens,
    epoch,
    output_dir,
    device,
    num_samples=3
):
    """解碼並儲存音頻樣本和頻譜圖"""
    samples_dir = Path(output_dir) / 'audio_samples' / f'epoch_{epoch}'
    samples_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, noisy_tokens.size(0))

    with torch.no_grad():
        for i in range(num_samples):
            if noisy_tokens.dim() == 3:
                noisy_tok = noisy_tokens[i].unsqueeze(0).to(device)
                pred_tok = pred_tokens[i].unsqueeze(0).to(device)
                clean_tok = clean_tokens[i].unsqueeze(0).to(device)
            else:
                noisy_tok = noisy_tokens[i].unsqueeze(0).unsqueeze(0).to(device)
                pred_tok = pred_tokens[i].unsqueeze(0).unsqueeze(0).to(device)
                clean_tok = clean_tokens[i].unsqueeze(0).unsqueeze(0).to(device)

            noisy_features = wavtokenizer.codes_to_features(noisy_tok)
            pred_features = wavtokenizer.codes_to_features(pred_tok)
            clean_features = wavtokenizer.codes_to_features(clean_tok)

            if noisy_features.dim() == 4:
                noisy_features = noisy_features.squeeze(2)
            if pred_features.dim() == 4:
                pred_features = pred_features.squeeze(2)
            if clean_features.dim() == 4:
                clean_features = clean_features.squeeze(2)

            bandwidth_id = torch.tensor([0], device=device)

            noisy_audio = wavtokenizer.decode(noisy_features, bandwidth_id=bandwidth_id).cpu()
            pred_audio = wavtokenizer.decode(pred_features, bandwidth_id=bandwidth_id).cpu()
            clean_audio = wavtokenizer.decode(clean_features, bandwidth_id=bandwidth_id).cpu()

            if noisy_audio.dim() == 1:
                noisy_audio = noisy_audio.unsqueeze(0)
            if pred_audio.dim() == 1:
                pred_audio = pred_audio.unsqueeze(0)
            if clean_audio.dim() == 1:
                clean_audio = clean_audio.unsqueeze(0)

            torchaudio.save(
                str(samples_dir / f'sample_{i}_noisy.wav'),
                noisy_audio.cpu(),
                24000
            )
            torchaudio.save(
                str(samples_dir / f'sample_{i}_predicted.wav'),
                pred_audio.cpu(),
                24000
            )
            torchaudio.save(
                str(samples_dir / f'sample_{i}_clean.wav'),
                clean_audio.cpu(),
                24000
            )

            plot_spectrograms(
                noisy_audio.cpu().squeeze().numpy(),
                pred_audio.cpu().squeeze().numpy(),
                clean_audio.cpu().squeeze().numpy(),
                str(samples_dir / f'sample_{i}_spectrogram.png')
            )


def plot_spectrograms(noisy_audio, pred_audio, clean_audio, save_path):
    """繪製三個音頻的頻譜圖"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, (audio, title) in enumerate([
        (noisy_audio, 'Noisy Audio'),
        (pred_audio, 'Predicted Audio'),
        (clean_audio, 'Clean Audio (Target)')
    ]):
        D = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

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


def plot_loss_curves(train_losses, val_losses, train_accs, val_accs,
                     train_speaker_losses, val_speaker_losses, output_path):
    """
    繪製訓練和驗證的損失及準確率曲線 (包含 Speaker Loss)
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Total Loss (CE + Speaker Loss)
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (CE + Speaker)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Token Accuracy
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Token Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Speaker L2 Loss
    axes[1, 0].plot(epochs, train_speaker_losses, 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, val_speaker_losses, 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Speaker L2 Loss')
    axes[1, 0].set_title('Speaker Embedding L2 Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Learning Rate (placeholder - can be filled later)
    axes[1, 1].text(0.5, 0.5, 'Reserved for\nLearning Rate',
                    ha='center', va='center', fontsize=14)
    axes[1, 1].set_title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


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
    """訓練一個 epoch (包含 Speaker Loss)"""
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_speaker_loss = 0.0
    total_correct = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (noisy_tokens, clean_tokens, content_ids) in enumerate(progress_bar):
        noisy_tokens = noisy_tokens.to(device)
        clean_tokens = clean_tokens.to(device)

        # Forward pass
        logits = model(noisy_tokens, return_logits=True)  # (B, T, 4096)

        # Compute loss (CE + Speaker Loss)
        loss, ce_loss, speaker_loss = criterion(
            logits, clean_tokens, noisy_tokens, current_epoch=epoch
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_speaker_loss += speaker_loss.item()

        # Token accuracy
        B, T, _ = logits.shape
        pred_tokens = logits.argmax(dim=-1)
        correct = (pred_tokens == clean_tokens).sum().item()
        total_correct += correct
        total_tokens += B * T

        # Update progress bar
        current_acc = (total_correct / total_tokens) * 100
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'spk': f'{speaker_loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_speaker_loss = total_speaker_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100

    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'speaker_loss': avg_speaker_loss,
        'accuracy': accuracy
    }


def validate_epoch(model, dataloader, criterion, device, epoch):
    """驗證一個 epoch (包含 Speaker Loss)"""
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_speaker_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for noisy_tokens, clean_tokens, content_ids in tqdm(dataloader, desc="Validation"):
            noisy_tokens = noisy_tokens.to(device)
            clean_tokens = clean_tokens.to(device)

            # Forward pass
            logits = model(noisy_tokens, return_logits=True)

            # Compute loss
            loss, ce_loss, speaker_loss = criterion(
                logits, clean_tokens, noisy_tokens, current_epoch=epoch
            )

            # Statistics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_speaker_loss += speaker_loss.item()

            # Token accuracy
            B, T, _ = logits.shape
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == clean_tokens).sum().item()
            total_correct += correct
            total_tokens += B * T

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_speaker_loss = total_speaker_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100

    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'speaker_loss': avg_speaker_loss,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='Token Denoising Transformer Training with Speaker Loss')

    # 數據參數
    parser.add_argument('--input_dirs', nargs='+', required=True, help='含噪音輸入目錄')
    parser.add_argument('--target_dir', required=True, help='乾淨目標目錄')
    parser.add_argument('--output_dir', default='./results/exp2/speaker_loss', help='輸出目錄')
    parser.add_argument('--max_sentences_per_speaker', type=int, default=None,
                       help='每個語者最多使用的句子數 (None=全部)')

    # 模型參數
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 維度')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN 維度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=600, help='訓練 epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')

    # Speaker Loss 參數
    parser.add_argument('--lambda_speaker', type=float, default=0.5,
                       help='Speaker loss 權重 (建議: 0.1, 0.5, 1.0)')
    parser.add_argument('--speaker_model_type', type=str, default='ecapa',
                       choices=['ecapa', 'resemblyzer'],
                       help='Speaker encoder 類型')
    parser.add_argument('--speaker_loss_start_epoch', type=int, default=0,
                       help='從第幾個 epoch 開始加入 speaker loss')
    parser.add_argument('--compute_speaker_every_n_steps', type=int, default=1,
                       help='每 N 步計算一次 speaker loss (用於加速訓練)')

    # WavTokenizer 參數
    parser.add_argument('--wavtokenizer_config',
                       default='/home/sbplab/ruizi/WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                       help='WavTokenizer 配置文件')
    parser.add_argument('--wavtokenizer_checkpoint',
                       default='/home/sbplab/ruizi/WavTokenizer/results/smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_epoch_1200.pth',
                       help='WavTokenizer checkpoint')

    args = parser.parse_args()

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'audio_samples').mkdir(parents=True, exist_ok=True)

    # 設置 logger
    logger = setup_logger(output_dir)
    logger.info("=" * 80)
    logger.info("Token Denoising Transformer 訓練 (Baseline + Speaker Loss)")
    logger.info("=" * 80)

    # 保存配置
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"配置已保存至: {config_path}")

    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")

    # 載入 WavTokenizer
    logger.info("載入 WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(
        args.wavtokenizer_config,
        args.wavtokenizer_checkpoint
    )
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()

    # 提取 Codebook
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    logger.info(f"Codebook 形狀: {codebook.shape}")

    # 創建數據集
    logger.info("創建數據集...")
    logger.info(f"輸入目錄: {args.input_dirs}")
    logger.info(f"目標目錄: {args.target_dir}")

    # Token collate function
    def token_collate_fn(batch):
        """Collate function with content IDs"""
        noisy_tokens_list = []
        clean_tokens_list = []
        content_ids_list = []

        for noisy_audio, clean_audio, content_id in batch:
            noisy_audio = noisy_audio.to(device).unsqueeze(0)
            clean_audio = clean_audio.to(device).unsqueeze(0)

            with torch.no_grad():
                _, noisy_tokens = wavtokenizer.encode_infer(
                    noisy_audio,
                    bandwidth_id=torch.tensor([0], device=device)
                )
                _, clean_tokens = wavtokenizer.encode_infer(
                    clean_audio,
                    bandwidth_id=torch.tensor([0], device=device)
                )

            noisy_tokens_list.append(noisy_tokens[0])
            clean_tokens_list.append(clean_tokens[0])
            content_ids_list.append(content_id)

        max_len = max(
            max(t.shape[1] for t in noisy_tokens_list),
            max(t.shape[1] for t in clean_tokens_list)
        )

        padded_noisy = []
        padded_clean = []

        for noisy_t, clean_t in zip(noisy_tokens_list, clean_tokens_list):
            curr_noisy = noisy_t.squeeze(0)
            if curr_noisy.shape[0] < max_len:
                pad_size = max_len - curr_noisy.shape[0]
                curr_noisy = torch.nn.functional.pad(curr_noisy, (0, pad_size), value=0)
            padded_noisy.append(curr_noisy)

            curr_clean = clean_t.squeeze(0)
            if curr_clean.shape[0] < max_len:
                pad_size = max_len - curr_clean.shape[0]
                curr_clean = torch.nn.functional.pad(curr_clean, (0, pad_size), value=0)
            padded_clean.append(curr_clean)

        noisy_tokens_batch = torch.stack(padded_noisy, dim=0)
        clean_tokens_batch = torch.stack(padded_clean, dim=0)

        numeric_ids = []
        for cid in content_ids_list:
            if isinstance(cid, str):
                digits = ''.join(c for c in cid if c.isdigit())
                numeric_ids.append(int(digits) if digits else hash(cid) % 1000)
            else:
                numeric_ids.append(int(cid))

        content_ids_batch = torch.tensor(numeric_ids, dtype=torch.long)

        return noisy_tokens_batch, clean_tokens_batch, content_ids_batch

    # 創建完整數據集
    audio_dataset = AudioDataset(
        input_dirs=args.input_dirs,
        target_dir=args.target_dir,
        max_sentences_per_speaker=args.max_sentences_per_speaker
    )

    logger.info(f"總數據集大小: {len(audio_dataset)} 個音頻對")

    # 按語者分割數據集
    val_speakers = ['girl9', 'girl10', 'boy7', 'boy8']
    train_indices = []
    val_indices = []

    for idx in range(len(audio_dataset)):
        filename = audio_dataset.paired_files[idx]['input']
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 2:
            speaker = parts[1]
        else:
            speaker = parts[0]

        if speaker in val_speakers:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    train_audio_dataset = torch.utils.data.Subset(audio_dataset, train_indices)
    val_audio_dataset = torch.utils.data.Subset(audio_dataset, val_indices)

    logger.info(f"訓練集大小: {len(train_audio_dataset)}")
    logger.info(f"驗證集大小: {len(val_audio_dataset)}")
    logger.info(f"驗證集語者 (未見語者): {val_speakers}")

    # DataLoader
    train_loader = DataLoader(
        train_audio_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=token_collate_fn
    )
    val_loader = DataLoader(
        val_audio_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=token_collate_fn
    )

    # 創建模型
    logger.info("=" * 80)
    logger.info("創建 Token Denoising Transformer")
    logger.info("=" * 80)
    model = TokenDenoisingTransformer(
        codebook=codebook,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可訓練參數: {trainable_params:,}")

    # 損失函數 (包含 Speaker Loss)
    logger.info("=" * 80)
    logger.info("損失函數配置")
    logger.info("=" * 80)
    logger.info(f"主任務: CrossEntropy Loss")
    logger.info(f"輔助約束: Speaker Embedding L2 Loss")
    logger.info(f"  - Speaker model: {args.speaker_model_type}")
    logger.info(f"  - Lambda (權重): {args.lambda_speaker}")
    logger.info(f"  - Start epoch: {args.speaker_loss_start_epoch}")
    logger.info(f"  - Compute every N steps: {args.compute_speaker_every_n_steps}")
    logger.info("=" * 80)

    criterion = create_loss_with_speaker(
        wavtokenizer=wavtokenizer,
        speaker_model_type=args.speaker_model_type,
        lambda_speaker=args.lambda_speaker,
        speaker_loss_start_epoch=args.speaker_loss_start_epoch,
        compute_speaker_every_n_steps=args.compute_speaker_every_n_steps,
        device=device
    )

    # 優化器
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True, min_lr=1e-6
    )

    # 訓練歷史記錄
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_speaker_loss_history = []
    val_speaker_loss_history = []

    # 訓練循環
    logger.info("開始訓練...")
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        # 訓練
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"CE: {train_metrics['ce_loss']:.4f}, "
                   f"Speaker: {train_metrics['speaker_loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.2f}%")

        # 驗證
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"CE: {val_metrics['ce_loss']:.4f}, "
                   f"Speaker: {val_metrics['speaker_loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.2f}%")

        # 記錄訓練歷史
        train_loss_history.append(train_metrics['loss'])
        val_loss_history.append(val_metrics['loss'])
        train_acc_history.append(train_metrics['accuracy'])
        val_acc_history.append(val_metrics['accuracy'])
        train_speaker_loss_history.append(train_metrics['speaker_loss'])
        val_speaker_loss_history.append(val_metrics['speaker_loss'])

        # 更新學習率
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning Rate: {current_lr:.2e}")

        # 保存 checkpoint
        if epoch % 100 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)

        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics
            }, best_model_path)
            logger.info(f"  ✓ 保存最佳模型 (Val Loss: {best_val_loss:.4f})")

        # 繪製損失曲線
        if epoch % 50 == 0 and epoch > 0:
            try:
                plot_path = output_dir / f'loss_curves_epoch_{epoch}.png'
                plot_loss_curves(
                    train_loss_history,
                    val_loss_history,
                    train_acc_history,
                    val_acc_history,
                    train_speaker_loss_history,
                    val_speaker_loss_history,
                    str(plot_path)
                )
                logger.info(f"  ✓ 已保存損失曲線")
            except Exception as e:
                logger.error(f"  繪製損失曲線時出錯: {e}")

        # 保存音頻樣本
        if epoch % 100 == 0 or epoch == args.num_epochs:
            logger.info(f"  保存音頻樣本...")
            try:
                model.eval()
                train_batch = next(iter(train_loader))
                noisy_tokens_batch, clean_tokens_batch, _ = train_batch

                with torch.no_grad():
                    pred_logits = model(noisy_tokens_batch[:3], return_logits=True)
                    pred_tokens = pred_logits.argmax(dim=-1)

                save_audio_samples(
                    wavtokenizer,
                    noisy_tokens_batch[:3],
                    pred_tokens,
                    clean_tokens_batch[:3],
                    epoch,
                    output_dir,
                    device,
                    num_samples=3
                )
                logger.info(f"  ✓ 已保存音頻樣本")
                model.train()
            except Exception as e:
                logger.error(f"  保存音頻樣本時出錯: {e}")

    logger.info("訓練完成！")
    logger.info(f"最佳驗證損失: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

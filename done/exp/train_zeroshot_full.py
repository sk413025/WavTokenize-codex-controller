"""
Zero-Shot Speaker Denoising Transformer 完整實驗

目的: 公平對比 baseline，驗證 zero-shot speaker conditioning 效果
配置:
- 18 speakers (14 train, 4 val) - 與 baseline 相同
- 每人最多 288 sentences - 與 baseline 相同
- 3 種材質 (box, papercup, plastic)
- 100 epochs
- Batch size 14 - 與 baseline 相同
- 預計 6-12 小時完成

使用方式:
    python train_zeroshot_full.py \\
        --input_dirs ../../data/raw/box ../../data/raw/papercup ../../data/raw/plastic \\
        --target_dir ../../data/clean/box2 \\
        --output_dir ./results/zeroshot_full \\
        --num_epochs 100 \\
        --batch_size 14 \\
        --max_sentences_per_speaker 288
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
# train_zeroshot_quick.py 在 done/exp/ 下
# 需要加入 done/exp/.. (done) 和 done/exp/../.. (c_code) 到 path
sys.path.insert(0, str(Path(__file__).parent.parent))  # 加入 done/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # 加入 c_code/

from decoder.pretrained import WavTokenizer
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 導入 zero-shot 模組
from model_zeroshot import ZeroShotDenoisingTransformer
from speaker_encoder import create_speaker_encoder
from data_zeroshot import ZeroShotAudioDataset, zeroshot_collate_fn_with_speaker


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
    axes[0].set_title('CrossEntropy Loss (Zero-Shot)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Token Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Token Accuracy (Zero-Shot)')
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

    使用 collate function 預處理版本：
    - DataLoader 的 collate function 已經完成 tokenization 和 speaker embedding 提取
    - 直接接收處理好的 tokens 和 embeddings
    - 更高效、避免重複計算

    Args:
        model: ZeroShotDenoisingTransformer
        dataloader: DataLoader (collate function 返回字典)
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

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # 從 collate function 接收已處理好的 batch (字典格式)
        noisy_tokens = batch['noisy_tokens'].to(device)
        clean_tokens = batch['clean_tokens'].to(device)
        speaker_embeddings = batch['speaker_embeddings'].to(device)
        content_ids = batch['content_ids']

        optimizer.zero_grad()

        # --- 模型前向傳播 (collate function 已經完成所有預處理) ---
        logits = model(noisy_tokens, speaker_embeddings, return_logits=True)  # (B, T, 4096)

        # --- 計算損失 ---
        B, T, vocab = logits.shape
        logits_flat = logits.reshape(B * T, vocab)
        clean_tokens_flat = clean_tokens.reshape(B * T)
        loss = criterion(logits_flat, clean_tokens_flat)

        # --- 反向傳播 ---
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
    """
    驗證一個 epoch

    使用 collate function 預處理版本：與 train_epoch 相同
    """
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # 從 collate function 接收已處理好的 batch
            noisy_tokens = batch['noisy_tokens'].to(device)
            clean_tokens = batch['clean_tokens'].to(device)
            speaker_embeddings = batch['speaker_embeddings'].to(device)
            content_ids = batch['content_ids']

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
    parser = argparse.ArgumentParser(description='Zero-Shot Speaker Denoising - Full Experiment')

    # 數據參數
    parser.add_argument('--input_dirs', nargs='+', required=True, help='含噪音輸入目錄')
    parser.add_argument('--target_dir', required=True, help='乾淨目標目錄')
    parser.add_argument('--output_dir', default='./results/zeroshot_full', help='輸出目錄')
    parser.add_argument('--max_sentences_per_speaker', type=int, default=288,
                       help='每個語者最多使用的句子數 (與 baseline 相同)')

    # 模型參數
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 維度')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN 維度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    # Speaker Encoder 參數
    parser.add_argument('--speaker_encoder', type=str, default='ecapa',
                       choices=['ecapa', 'resemblyzer'], help='Speaker encoder 類型')
    parser.add_argument('--speaker_dim', type=int, default=256, help='Speaker embedding 維度')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=14, help='Batch size (與 baseline 相同)')
    parser.add_argument('--num_epochs', type=int, default=100, help='訓練 epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')

    # WavTokenizer 參數 (使用與 baseline 相同的配置)
    parser.add_argument('--wavtokenizer_config',
                       default='../../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                       help='WavTokenizer 配置文件')
    parser.add_argument('--wavtokenizer_checkpoint',
                       default='../../models/wavtokenizer_large_speech_320_24k.ckpt',
                       help='WavTokenizer checkpoint')

    args = parser.parse_args()

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'audio_samples').mkdir(parents=True, exist_ok=True)

    # 設置 logger
    logger = setup_logger(output_dir)
    logger.info("=" * 80)
    logger.info("Zero-Shot Speaker Denoising Transformer - 完整實驗")
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
    logger.info(f"Codebook 形狀: {codebook.shape}")  # (4096, 512)

    # 創建 Speaker Encoder
    logger.info(f"創建 Speaker Encoder ({args.speaker_encoder})...")
    speaker_encoder = create_speaker_encoder(
        model_type=args.speaker_encoder,
        freeze=True,
        output_dim=args.speaker_dim
    )
    speaker_encoder.eval()
    logger.info(f"Speaker Encoder 輸出維度: {args.speaker_dim}")

    # 創建數據集（使用所有 18 speakers）
    logger.info("創建數據集...")
    logger.info(f"輸入目錄: {args.input_dirs}")
    logger.info(f"目標目錄: {args.target_dir}")
    logger.info(f"每個語者句子數: {args.max_sentences_per_speaker}")

    # 創建完整數據集（使用所有 speakers，與 baseline 相同）
    audio_dataset = ZeroShotAudioDataset(
        input_dirs=args.input_dirs,
        target_dir=args.target_dir,
        max_sentences_per_speaker=args.max_sentences_per_speaker
        # 不限制 speakers，使用所有 18 個
    )

    logger.info(f"總數據集大小: {len(audio_dataset)} 個音頻對")

    # 數據集分割（與 baseline 相同的分割方式）
    # 驗證集語者：girl9, girl10, boy7, boy8
    # 訓練集語者：其他所有語者
    val_speakers = ['girl9', 'girl10', 'boy7', 'boy8']
    train_speakers = ['boy1', 'boy3', 'boy4', 'boy5', 'boy6', 'boy9', 'boy10',
                     'girl2', 'girl3', 'girl4', 'girl6', 'girl7', 'girl8', 'girl11']

    logger.info("=" * 80)
    logger.info("數據集分割 (與 baseline 相同)")
    logger.info("=" * 80)
    logger.info(f"驗證集語者: {val_speakers}")
    logger.info(f"訓練集語者: {train_speakers}")

    train_indices = []
    val_indices = []

    for idx in range(len(audio_dataset)):
        filename = audio_dataset.paired_files[idx]['input']
        parts = os.path.basename(filename).split('_')
        speaker = parts[1] if len(parts) >= 2 else 'unknown'

        if speaker in val_speakers:
            val_indices.append(idx)
        elif speaker in train_speakers:
            train_indices.append(idx)
        # 忽略不在列表中的 speakers

    train_audio_dataset = torch.utils.data.Subset(audio_dataset, train_indices)
    val_audio_dataset = torch.utils.data.Subset(audio_dataset, val_indices)

    logger.info(f"訓練集大小: {len(train_audio_dataset)} 樣本 (14 speakers)")
    logger.info(f"驗證集大小: {len(val_audio_dataset)} 樣本 (4 speakers)")
    logger.info(f"訓練/驗證比例: {len(train_audio_dataset)/(len(train_audio_dataset)+len(val_audio_dataset))*100:.1f}% / {len(val_audio_dataset)/(len(train_audio_dataset)+len(val_audio_dataset))*100:.1f}%")

    # 創建 collate function wrapper
    from functools import partial
    token_collate_fn = partial(
        zeroshot_collate_fn_with_speaker,
        wavtokenizer=wavtokenizer,
        speaker_encoder=speaker_encoder,
        device=device
    )

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

    # 創建 Zero-Shot 模型
    logger.info("=" * 80)
    logger.info("創建 Zero-Shot Denoising Transformer")
    logger.info("=" * 80)
    model = ZeroShotDenoisingTransformer(
        codebook=codebook,
        speaker_embed_dim=args.speaker_dim,  # 正確的參數名
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    # 計算參數量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_in_model = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # 計算 buffers (Codebook)
    buffer_params = sum(b.numel() for b in model.buffers())

    # 計算 Speaker Encoder 參數
    speaker_encoder_params = sum(p.numel() for p in speaker_encoder.parameters())

    # 總凍結參數 = buffers + speaker encoder
    total_frozen = buffer_params + speaker_encoder_params

    # 總參數 = 可訓練 + 凍結
    total_params = trainable_params + total_frozen

    logger.info("模型架構:")
    logger.info(f"  - d_model: {args.d_model}")
    logger.info(f"  - nhead: {args.nhead}")
    logger.info(f"  - num_layers: {args.num_layers}")
    logger.info(f"  - speaker_dim: {args.speaker_dim}")
    logger.info(f"  - fusion_type: additive")
    logger.info("")
    logger.info(f"模型總參數數量: {total_params:,}")
    logger.info(f"  - 可訓練參數: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"  - 凍結參數: {total_frozen:,} ({total_frozen/total_params*100:.2f}%)")
    logger.info(f"    - Codebook (buffer): {buffer_params:,}")
    logger.info(f"    - ECAPA-TDNN (外部): {speaker_encoder_params:,}")
    logger.info("=" * 80)

    # 損失函數
    criterion = nn.CrossEntropyLoss()

    # 優化器
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6
    )

    # 訓練歷史記錄
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # 訓練循環
    logger.info("開始訓練...")
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

        # 更新學習率
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning Rate: {current_lr:.2e}")

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

    # 最終總結
    logger.info("=" * 80)
    logger.info("快速驗證實驗完成！")
    logger.info("=" * 80)
    logger.info(f"最佳驗證準確率: {best_val_acc:.2f}%")
    logger.info(f"最佳驗證損失: {best_val_loss:.4f}")
    logger.info("")
    logger.info("結果判斷:")

    baseline_acc = 38.0  # Baseline accuracy

    if best_val_acc > 45.0:
        logger.info(f"  ✅ Val Acc {best_val_acc:.2f}% > 45% - 架構有效！")
        logger.info(f"  ✅ 相比 Baseline ({baseline_acc}%)，提升了 {best_val_acc - baseline_acc:.2f}%")
        logger.info("  ✅ 建議：擴大規模到完整數據集（14 speakers）")
    elif best_val_acc > 40.0:
        logger.info(f"  ⚠️  Val Acc {best_val_acc:.2f}% 在 40-45% - 勉強及格")
        logger.info(f"  ⚠️  相比 Baseline ({baseline_acc}%)，略有提升")
        logger.info("  ⚠️  建議：嘗試改進 fusion 策略或 fine-tune ECAPA")
    else:
        logger.info(f"  ❌ Val Acc {best_val_acc:.2f}% < 40% - 架構可能無效")
        logger.info(f"  ❌ 未能超越 Baseline ({baseline_acc}%)")
        logger.info("  ❌ 建議：重新檢查 speaker conditioning 實現")

    # 繪製損失曲線
    logger.info("")
    logger.info("=" * 80)
    logger.info("生成可視化結果")
    logger.info("=" * 80)

    plot_path = output_dir / 'loss_curves.png'
    plot_loss_curves(
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        str(plot_path)
    )
    logger.info(f"✓ 損失曲線已保存: {plot_path.name}")

    # 生成音檔樣本和頻譜圖（從驗證集）
    logger.info("")
    logger.info("生成音檔樣本和頻譜圖...")
    try:
        model.eval()

        # 從驗證集取一個 batch
        val_batch = next(iter(val_loader))
        noisy_tokens_batch = val_batch['noisy_tokens'].to(device)
        clean_tokens_batch = val_batch['clean_tokens'].to(device)
        speaker_embeddings_batch = val_batch['speaker_embeddings'].to(device)

        # 取前 3 個樣本
        num_samples = min(3, noisy_tokens_batch.shape[0])

        samples_dir = output_dir / 'audio_samples'
        samples_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            noisy_tok = noisy_tokens_batch[i:i+1]  # (1, T)
            clean_tok = clean_tokens_batch[i:i+1]  # (1, T)
            speaker_emb = speaker_embeddings_batch[i:i+1]  # (1, speaker_dim)

            # 預測
            with torch.no_grad():
                pred_logits = model(noisy_tok, speaker_emb, return_logits=True)  # (1, T, vocab)
                pred_tok = pred_logits.argmax(dim=-1)  # (1, T)

            # 解碼為音頻
            with torch.no_grad():
                # 將 (1, T) -> (1, 1, T) for codes_to_features
                noisy_tok_3d = noisy_tok.unsqueeze(1)
                pred_tok_3d = pred_tok.unsqueeze(1)
                clean_tok_3d = clean_tok.unsqueeze(1)

                noisy_features = wavtokenizer.codes_to_features(noisy_tok_3d)
                pred_features = wavtokenizer.codes_to_features(pred_tok_3d)
                clean_features = wavtokenizer.codes_to_features(clean_tok_3d)

                # Squeeze if needed
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

                # Ensure 2D (C, T)
                if noisy_audio.dim() == 1:
                    noisy_audio = noisy_audio.unsqueeze(0)
                if pred_audio.dim() == 1:
                    pred_audio = pred_audio.unsqueeze(0)
                if clean_audio.dim() == 1:
                    clean_audio = clean_audio.unsqueeze(0)

            # 保存音頻
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

            # 繪製頻譜圖
            plot_spectrograms(
                noisy_audio.squeeze(0).numpy(),
                pred_audio.squeeze(0).numpy(),
                clean_audio.squeeze(0).numpy(),
                str(samples_dir / f'sample_{i}_spectrogram.png')
            )

            logger.info(f"  ✓ 樣本 {i}: 音檔和頻譜圖已保存")

        logger.info(f"✓ 所有音檔樣本已保存至: {samples_dir}")

    except Exception as e:
        logger.error(f"生成音檔樣本時出錯: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 所有結果已保存至: {output_dir}")
    logger.info("=" * 80)
    logger.info("生成的文件:")
    logger.info(f"  - training.log: 訓練日誌")
    logger.info(f"  - loss_curves.png: 損失和準確率曲線")
    logger.info(f"  - best_model.pth: 最佳模型")
    logger.info(f"  - audio_samples/sample_*_noisy.wav: 噪音音檔")
    logger.info(f"  - audio_samples/sample_*_predicted.wav: 預測音檔")
    logger.info(f"  - audio_samples/sample_*_clean.wav: 乾淨音檔")
    logger.info(f"  - audio_samples/sample_*_spectrogram.png: 頻譜圖")


if __name__ == '__main__':
    main()

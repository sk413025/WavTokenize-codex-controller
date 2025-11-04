"""
Zero-Shot Speaker Denoising Transformer - 完整實驗（使用緩存版本 + 分析功能）

新增分析功能：
1. Token 預測分布分析：檢測是否總是預測同一種 token
2. Speaker Embedding 記錄：保存所有語者的 embedding
3. 自動生成視覺化：t-SNE 降維看語者分布

從 train_zeroshot_full_cached.py 修改而來
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
import numpy as np
from collections import Counter

# 添加必要的路徑
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from decoder.pretrained import WavTokenizer
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 導入 zero-shot 模組
from model_zeroshot import ZeroShotDenoisingTransformer
from data_zeroshot import ZeroShotAudioDatasetCached, cached_collate_fn


def analyze_token_predictions(pred_tokens, top_k=20):
    """
    分析 token 預測分布

    Args:
        pred_tokens: (B, T) 預測的 token IDs
        top_k: 顯示最常見的前 K 個 token

    Returns:
        dict: 統計信息
    """
    # 展平所有 token
    all_tokens = pred_tokens.cpu().numpy().flatten()

    # 統計
    counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(counter)

    # 最常見的 token
    most_common = counter.most_common(top_k)

    # 計算熵（多樣性指標）
    probs = np.array([count / total_tokens for count in counter.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # 計算 top-1 token 佔比
    top1_ratio = most_common[0][1] / total_tokens if most_common else 0

    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'unique_ratio': unique_tokens / 4096,  # 4096 是總 vocab size
        'entropy': entropy,
        'top1_token': most_common[0][0] if most_common else None,
        'top1_ratio': top1_ratio,
        'most_common': most_common[:10]  # 只保存前10個
    }


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


def plot_loss_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    """繪製訓練和驗證的損失及準確率曲線"""
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. CE Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CE Loss')
    axes[0].set_title('CrossEntropy Loss (Zero-Shot Cached)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Token Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    axes[1].axhline(y=38.19, color='gray', linestyle='--', label='Baseline (38.19%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Token Accuracy (%)')
    axes[1].set_title('Token Accuracy (Zero-Shot Cached)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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


def collect_speaker_embeddings(dataset, device, max_samples_per_speaker=20):
    """
    收集所有 speaker embeddings（按語者分組）

    Args:
        dataset: ZeroShotAudioDatasetCached
        device: torch device
        max_samples_per_speaker: 每位語者最多收集多少個樣本

    Returns:
        embeddings: (N, D) numpy array
        speaker_ids: list of speaker IDs (e.g., 'girl1', 'boy2')
    """
    from collections import defaultdict

    embeddings_by_speaker = defaultdict(list)

    # 遍歷數據集收集 speaker embeddings
    for idx in tqdm(range(len(dataset)), desc="Collecting speaker embeddings"):
        sample = dataset[idx]
        # Cached dataset 使用 'speaker_embedding' (單數)
        speaker_emb_tensor = sample.get('speaker_embedding', sample.get('speaker_embeddings'))
        speaker_emb = speaker_emb_tensor.cpu().numpy() if isinstance(speaker_emb_tensor, torch.Tensor) else speaker_emb_tensor

        # 從 content_id 中提取 speaker ID
        # content_id 格式: "ID_speaker_sentence" 如 "1_girl1_7"
        content_id = sample.get('content_id', '')
        if isinstance(content_id, (int, torch.Tensor)):
            content_id = str(int(content_id))

        # 解析 content_id 獲取 speaker
        parts = content_id.split('_')
        if len(parts) >= 2:
            speaker_id = parts[1]  # 如 "girl1", "boy7"
        else:
            # Fallback: 使用 embedding 哈希
            emb_hash = hash(tuple(speaker_emb.flatten()[:10]))
            speaker_id = f"speaker_{abs(emb_hash) % 18}"

        if len(embeddings_by_speaker[speaker_id]) < max_samples_per_speaker:
            embeddings_by_speaker[speaker_id].append(speaker_emb)

    # 整理成列表
    embeddings_list = []
    speaker_ids = []

    for speaker_id, embs in sorted(embeddings_by_speaker.items()):
        for emb in embs:
            embeddings_list.append(emb)
            speaker_ids.append(speaker_id)

    embeddings = np.array(embeddings_list)
    return embeddings, speaker_ids


def visualize_speaker_embeddings(train_embeddings, val_embeddings, train_speaker_ids, val_speaker_ids, save_path):
    """
    使用 t-SNE 視覺化 speaker embeddings（帶語者編號標籤）

    Args:
        train_embeddings: (N_train, D) 訓練集 embeddings
        val_embeddings: (N_val, D) 驗證集 embeddings
        train_speaker_ids: list of train speaker IDs
        val_speaker_ids: list of val speaker IDs
        save_path: 保存路徑
    """
    from sklearn.manifold import TSNE

    # 合併所有 embeddings
    all_embeddings = np.concatenate([train_embeddings, val_embeddings], axis=0)
    all_speaker_ids = train_speaker_ids + val_speaker_ids

    # t-SNE 降維
    logger = logging.getLogger(__name__)
    logger.info("執行 t-SNE 降維...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)//5))
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # 分離訓練集和驗證集的座標
    train_2d = embeddings_2d[:len(train_embeddings)]
    val_2d = embeddings_2d[len(train_embeddings):]

    # 為每個語者計算中心點（用於標註）
    unique_train_speakers = sorted(set(train_speaker_ids))
    unique_val_speakers = sorted(set(val_speaker_ids))

    speaker_centers = {}

    # 計算訓練集語者中心
    for speaker in unique_train_speakers:
        indices = [i for i, sid in enumerate(train_speaker_ids) if sid == speaker]
        if indices:
            center = train_2d[indices].mean(axis=0)
            speaker_centers[speaker] = (center, 'train')

    # 計算驗證集語者中心
    for speaker in unique_val_speakers:
        indices = [i for i, sid in enumerate(val_speaker_ids) if sid == speaker]
        if indices:
            center = val_2d[indices].mean(axis=0)
            speaker_centers[speaker] = (center, 'val')

    # 繪圖
    fig, ax = plt.subplots(figsize=(14, 10))

    # 繪製訓練集點
    ax.scatter(train_2d[:, 0], train_2d[:, 1],
                c='blue', alpha=0.5, s=30, label=f'Train ({len(unique_train_speakers)} speakers)',
                edgecolors='darkblue', linewidth=0.5)

    # 繪製驗證集點
    ax.scatter(val_2d[:, 0], val_2d[:, 1],
                c='red', alpha=0.5, s=30, label=f'Val ({len(unique_val_speakers)} speakers)',
                edgecolors='darkred', linewidth=0.5)

    # 標註每個語者的編號
    for speaker, (center, split) in speaker_centers.items():
        # 提取語者編號（假設格式是 'speaker_N' 或 'girlN'/'boyN'）
        if 'girl' in speaker:
            label = speaker.replace('girl', 'G')
        elif 'boy' in speaker:
            label = speaker.replace('boy', 'B')
        elif 'speaker_' in speaker:
            label = speaker.replace('speaker_', 'S')
        else:
            label = speaker

        color = 'darkblue' if split == 'train' else 'darkred'
        fontweight = 'bold' if split == 'val' else 'normal'

        # 在中心位置標註
        ax.annotate(label,
                    xy=center,
                    fontsize=10,
                    fontweight=fontweight,
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white' if split == 'train' else 'lightyellow',
                             edgecolor=color,
                             alpha=0.8))

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Speaker Embedding Distribution (t-SNE)\n已知語者(藍) vs 未知語者(紅)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.2)

    # 添加說明文字
    textstr = f'總語者數: {len(speaker_centers)}\n訓練集: {len(unique_train_speakers)} 位\n驗證集: {len(unique_val_speakers)} 位'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ t-SNE 視覺化完成")
    logger.info(f"  - 訓練集語者: {unique_train_speakers}")
    logger.info(f"  - 驗證集語者: {unique_val_speakers}")


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, logger):
    """訓練一個 epoch（增加 token 分析）"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    all_predictions = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
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

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item()

        # Token accuracy
        pred_tokens = logits.argmax(dim=-1)
        correct = (pred_tokens == clean_tokens).sum().item()
        total_correct += correct
        total_tokens += B * T

        # 收集預測用於分析
        all_predictions.append(pred_tokens.detach())

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
    logger.info(f"    - 唯一 token 數: {token_stats['unique_tokens']}/4096 ({token_stats['unique_ratio']*100:.2f}%)")
    logger.info(f"    - 預測熵 (多樣性): {token_stats['entropy']:.4f}")
    logger.info(f"    - 最常見 token: {token_stats['top1_token']} (佔比 {token_stats['top1_ratio']*100:.2f}%)")
    if token_stats['top1_ratio'] > 0.5:
        logger.warning(f"    ⚠️  警告: >50% 的預測都是同一個 token!")

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'token_stats': token_stats
    }


def validate_epoch(model, dataloader, criterion, device, epoch, logger):
    """驗證一個 epoch（增加 token 分析）"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
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

            # 收集預測
            all_predictions.append(pred_tokens)

    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100

    # 分析 token 預測分布
    all_predictions = torch.cat(all_predictions, dim=0)
    token_stats = analyze_token_predictions(all_predictions)

    # 記錄到 log
    logger.info(f"  Token 預測分析 (Validation):")
    logger.info(f"    - 唯一 token 數: {token_stats['unique_tokens']}/4096 ({token_stats['unique_ratio']*100:.2f}%)")
    logger.info(f"    - 預測熵: {token_stats['entropy']:.4f}")
    logger.info(f"    - 最常見 token: {token_stats['top1_token']} (佔比 {token_stats['top1_ratio']*100:.2f}%)")

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'token_stats': token_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Speaker Denoising - Cached Training with Analysis')

    # 緩存參數
    parser.add_argument('--cache_dir', default='./data', help='緩存目錄')
    parser.add_argument('--output_dir', default='./results/zeroshot_full_cached', help='輸出目錄')

    # 模型參數
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 維度')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN 維度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=28, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='訓練 epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # 分析參數 ⭐ 新增
    parser.add_argument('--analyze_speakers', action='store_true', help='分析並視覺化 speaker embeddings')
    parser.add_argument('--speaker_analysis_freq', type=int, default=50, help='每 N epochs 分析一次 speaker embeddings')

    # WavTokenizer 參數
    parser.add_argument('--wavtokenizer_config',
                       default='../../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                       help='WavTokenizer 配置文件')
    parser.add_argument('--wavtokenizer_checkpoint',
                       default='../../models/wavtokenizer_large_speech_320_24k.ckpt',
                       help='WavTokenizer checkpoint')

    args = parser.parse_args()

    # 檢查緩存
    cache_dir = Path(args.cache_dir)
    train_cache_path = cache_dir / 'train_cache.pt'
    val_cache_path = cache_dir / 'val_cache.pt'
    config_cache_path = cache_dir / 'cache_config.pt'

    if not train_cache_path.exists():
        raise FileNotFoundError(f"訓練集緩存不存在: {train_cache_path}")
    if not val_cache_path.exists():
        raise FileNotFoundError(f"驗證集緩存不存在: {val_cache_path}")

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'audio_samples').mkdir(parents=True, exist_ok=True)
    (output_dir / 'speaker_analysis').mkdir(parents=True, exist_ok=True)  # ⭐ 新增

    # 設置 logger
    logger = setup_logger(output_dir)
    logger.info("=" * 80)
    logger.info("Zero-Shot Speaker Denoising Transformer - 完整實驗 + 分析")
    logger.info("=" * 80)

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

    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    logger.info(f"Codebook 形狀: {codebook.shape}")

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

    # 創建模型
    logger.info("=" * 80)
    logger.info("創建 Zero-Shot Denoising Transformer")
    logger.info("=" * 80)
    model = ZeroShotDenoisingTransformer(
        codebook=codebook,
        speaker_embed_dim=speaker_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可訓練參數: {trainable_params:,}")
    logger.info("=" * 80)

    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # 訓練歷史
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    token_diversity_history = []  # ⭐ 新增：記錄 token 多樣性

    best_val_loss = float('inf')
    best_val_acc = 0.0

    # ⭐ Speaker embedding 分析（訓練前）
    if args.analyze_speakers:
        logger.info("=" * 80)
        logger.info("收集 Speaker Embeddings...")
        logger.info("=" * 80)

        train_embeddings, train_speaker_ids = collect_speaker_embeddings(train_dataset, device, max_samples_per_speaker=20)
        val_embeddings, val_speaker_ids = collect_speaker_embeddings(val_dataset, device, max_samples_per_speaker=20)

        logger.info(f"訓練集 embeddings: {train_embeddings.shape}")
        logger.info(f"  - 語者數: {len(set(train_speaker_ids))}")
        logger.info(f"驗證集 embeddings: {val_embeddings.shape}")
        logger.info(f"  - 語者數: {len(set(val_speaker_ids))}")

        # 保存 embeddings 和 IDs
        np.save(output_dir / 'speaker_analysis' / 'train_embeddings.npy', train_embeddings)
        np.save(output_dir / 'speaker_analysis' / 'val_embeddings.npy', val_embeddings)

        # 保存 speaker IDs
        with open(output_dir / 'speaker_analysis' / 'train_speaker_ids.txt', 'w') as f:
            for sid in train_speaker_ids:
                f.write(f"{sid}\n")
        with open(output_dir / 'speaker_analysis' / 'val_speaker_ids.txt', 'w') as f:
            for sid in val_speaker_ids:
                f.write(f"{sid}\n")

        # 視覺化
        logger.info("生成 t-SNE 視覺化...")
        visualize_speaker_embeddings(
            train_embeddings,
            val_embeddings,
            train_speaker_ids,
            val_speaker_ids,
            output_dir / 'speaker_analysis' / 'speaker_distribution_tsne.png'
        )
        logger.info(f"✓ Speaker 分布圖已保存")

    # 訓練循環
    logger.info("開始訓練...")
    logger.info("")

    for epoch in range(1, args.num_epochs + 1):
        # 訓練
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )

        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.2f}%")

        # 驗證
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, logger
        )

        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.2f}%")

        # 記錄訓練歷史
        train_loss_history.append(train_metrics['loss'])
        val_loss_history.append(val_metrics['loss'])
        train_acc_history.append(train_metrics['accuracy'])
        val_acc_history.append(val_metrics['accuracy'])
        token_diversity_history.append({
            'epoch': epoch,
            'train_entropy': train_metrics['token_stats']['entropy'],
            'val_entropy': val_metrics['token_stats']['entropy'],
            'train_unique_ratio': train_metrics['token_stats']['unique_ratio'],
            'val_unique_ratio': val_metrics['token_stats']['unique_ratio']
        })

        # 更新學習率
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning Rate: {current_lr:.2e}")

        # 保存 checkpoint (每 50 epochs)
        if epoch % 50 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
                'token_diversity_history': token_diversity_history  # ⭐ 保存多樣性歷史
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

        # 每 50 epochs 繪製損失曲線
        if epoch % 50 == 0:
            plot_path = output_dir / f'loss_curves_epoch_{epoch}.png'
            plot_loss_curves(
                train_loss_history,
                val_loss_history,
                train_acc_history,
                val_acc_history,
                str(plot_path)
            )
            logger.info(f"  ✓ 已保存損失曲線")

    # 最終總結
    logger.info("=" * 80)
    logger.info("完整實驗完成！")
    logger.info("=" * 80)
    logger.info(f"最佳驗證準確率: {best_val_acc:.2f}%")

    # 保存 token 多樣性歷史
    diversity_path = output_dir / 'token_diversity_history.json'
    with open(diversity_path, 'w') as f:
        json.dump(token_diversity_history, f, indent=2)
    logger.info(f"✓ Token 多樣性歷史已保存: {diversity_path.name}")


if __name__ == '__main__':
    main()

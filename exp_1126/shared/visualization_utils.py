"""
Exp5 可視化工具：用於保存訓練過程中的頻譜圖、音檔和 loss 圖表

功能:
- 每 N epochs 保存訓練和驗證樣本的頻譜圖
- 每 N epochs 保存訓練和驗證樣本的重建音檔
- 繪製和保存訓練/驗證 loss 曲線
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import soundfile as sf
import logging

logger = logging.getLogger(__name__)


def save_loss_plot(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    output_path: Path,
    title: str = "Training Progress"
):
    """
    繪製並保存 loss 和 accuracy 曲線

    Args:
        train_losses: 訓練 loss 列表
        val_losses: 驗證 loss 列表
        train_accs: 訓練 accuracy 列表
        val_accs: 驗證 accuracy 列表
        output_path: 保存路徑
        title: 圖表標題
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add best val acc marker
    best_epoch = np.argmax(val_accs) + 1
    best_acc = max(val_accs)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax2.text(best_epoch, best_acc, f'Best: {best_acc:.2f}%\nEpoch {best_epoch}',
             fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved loss plot to {output_path}")


def compute_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> np.ndarray:
    """
    計算梅爾頻譜圖

    Args:
        audio: 音頻張量 [batch, time] 或 [time]
        sample_rate: 採樣率
        n_fft: FFT 窗口大小
        hop_length: 跳躍長度
        n_mels: 梅爾濾波器數量

    Returns:
        梅爾頻譜圖 [n_mels, time]
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed, cannot compute mel spectrogram")
        return None

    # 確保是 numpy array
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    # 確保是 1D
    if audio.ndim > 1:
        audio = audio.squeeze()

    # 計算梅爾頻譜圖
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # 轉為 dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def save_spectrogram_comparison(
    noisy_audio: torch.Tensor,
    clean_audio: torch.Tensor,
    reconstructed_audio: torch.Tensor,
    output_path: Path,
    sample_rate: int = 24000,
    title: str = "Spectrogram Comparison"
):
    """
    保存頻譜圖比較 (noisy vs clean vs reconstructed)

    Args:
        noisy_audio: 噪聲音頻 [batch, time] 或 [time]
        clean_audio: 乾淨音頻 [batch, time] 或 [time]
        reconstructed_audio: 重建音頻 [batch, time] 或 [time]
        output_path: 保存路徑
        sample_rate: 採樣率
        title: 圖表標題
    """
    # 計算頻譜圖
    noisy_mel = compute_mel_spectrogram(noisy_audio, sample_rate)
    clean_mel = compute_mel_spectrogram(clean_audio, sample_rate)
    recon_mel = compute_mel_spectrogram(reconstructed_audio, sample_rate)

    if noisy_mel is None:
        logger.warning("Cannot save spectrogram without librosa")
        return

    # 繪圖
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Noisy
    im1 = axes[0].imshow(noisy_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Noisy Audio', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mel Frequency', fontsize=10)
    plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')

    # Clean (Target)
    im2 = axes[1].imshow(clean_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Clean Audio (Target)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mel Frequency', fontsize=10)
    plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')

    # Reconstructed
    im3 = axes[2].imshow(recon_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Reconstructed Audio', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Mel Frequency', fontsize=10)
    axes[2].set_xlabel('Time Frame', fontsize=10)
    plt.colorbar(im3, ax=axes[2], format='%+2.0f dB')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved spectrogram to {output_path}")


def save_audio_samples(
    model: nn.Module,
    dataloader,
    device: str,
    output_dir: Path,
    epoch: int,
    num_samples: int = 3,
    sample_rate: int = 24000,
    prefix: str = "train"
):
    """
    保存音頻樣本 (noisy, clean, reconstructed) 和頻譜圖

    Args:
        model: 訓練的模型
        dataloader: 數據加載器
        device: 設備
        output_dir: 輸出目錄
        epoch: 當前 epoch
        num_samples: 保存樣本數量
        sample_rate: 採樣率
        prefix: 文件前綴 (train/val)
    """
    model.eval()

    # 創建子目錄（符合參考格式：audio_samples/epoch_X_training/）
    sample_dir = output_dir / "audio_samples" / f"epoch_{epoch}_{prefix}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if saved_count >= num_samples:
                break

            noisy_tokens = batch['noisy_tokens'].to(device)
            clean_tokens = batch['clean_tokens'].to(device)
            speaker_embeddings = batch['speaker_embeddings'].to(device)

            # 獲取音頻 (如果有的話)
            noisy_audio = batch.get('noisy_audio', None)
            clean_audio = batch.get('clean_audio', None)

            # 如果音頻不在 batch 中，嘗試從路徑載入
            if noisy_audio is None or clean_audio is None:
                noisy_path = batch.get('noisy_path', None)
                clean_path = batch.get('clean_path', None)

                if noisy_path is not None and clean_path is not None:
                    try:
                        import soundfile as sf_load
                        import os

                        # 音頻基礎目錄
                        noisy_base_dir = '/home/sbplab/ruizi/WavTokenize/data/raw/box'
                        clean_base_dir = '/home/sbplab/ruizi/WavTokenize/data/clean/box2'

                        # 輔助函數：構建完整路徑
                        def build_full_path(path):
                            if os.path.isabs(path) and os.path.exists(path):
                                return path
                            filename = os.path.basename(path)
                            if 'clean' in filename:
                                return os.path.join(clean_base_dir, filename)
                            else:
                                return os.path.join(noisy_base_dir, filename)

                        # 載入音頻 (處理批次 - 使用 pad_sequence 而不是 stack)
                        if isinstance(noisy_path, (list, tuple)):
                            noisy_audio_list = []
                            clean_audio_list = []
                            for n_path, c_path in zip(noisy_path, clean_path):
                                n_full_path = build_full_path(n_path)
                                c_full_path = build_full_path(c_path)
                                n_audio, _ = sf_load.read(n_full_path)
                                c_audio, _ = sf_load.read(c_full_path)
                                noisy_audio_list.append(torch.tensor(n_audio, dtype=torch.float32))
                                clean_audio_list.append(torch.tensor(c_audio, dtype=torch.float32))
                            # 使用 pad_sequence 處理長度不同的音頻
                            from torch.nn.utils.rnn import pad_sequence
                            noisy_audio = pad_sequence(noisy_audio_list, batch_first=True)
                            clean_audio = pad_sequence(clean_audio_list, batch_first=True)
                        else:
                            # 單個樣本
                            n_full_path = build_full_path(noisy_path)
                            c_full_path = build_full_path(clean_path)
                            n_audio, _ = sf_load.read(n_full_path)
                            c_audio, _ = sf_load.read(c_full_path)
                            noisy_audio = torch.tensor(n_audio, dtype=torch.float32).unsqueeze(0)
                            clean_audio = torch.tensor(c_audio, dtype=torch.float32).unsqueeze(0)
                    except Exception as e:
                        logger.warning(f"Batch {batch_idx} failed to load audio from paths: {str(e)}")
                        continue
                else:
                    logger.warning(f"Batch {batch_idx} missing audio and paths, skipping")
                    continue

            # 逐個樣本處理（避免長度不同的問題）
            batch_size = noisy_tokens.shape[0]
            for sample_idx in range(min(batch_size, num_samples - saved_count)):
                try:
                    # 獲取單個樣本
                    single_noisy_token = noisy_tokens[sample_idx:sample_idx+1]
                    single_speaker_emb = speaker_embeddings[sample_idx:sample_idx+1]

                    # 前向傳播得到預測
                    predicted_token = model(single_noisy_token, single_speaker_emb, return_logits=False)

                    # 獲取音頻
                    if isinstance(noisy_path, (list, tuple)) and len(noisy_path) > sample_idx:
                        sample_noisy_audio = noisy_audio[sample_idx] if noisy_audio.ndim > 1 else noisy_audio
                        sample_clean_audio = clean_audio[sample_idx] if clean_audio.ndim > 1 else clean_audio
                    else:
                        sample_noisy_audio = noisy_audio[sample_idx] if noisy_audio.ndim > 1 else noisy_audio
                        sample_clean_audio = clean_audio[sample_idx] if clean_audio.ndim > 1 else clean_audio

                    # 重建音頻 (使用 WavTokenizer decoder)
                    wavtokenizer = model.wavtokenizer if hasattr(model, 'wavtokenizer') else None
                    if wavtokenizer is None:
                        logger.warning("Model has no wavtokenizer attribute")
                        continue

                    # 重建音頻
                    with torch.no_grad():
                        reconstructed_audio = wavtokenizer.decode(predicted_token)  # [1, 1, T]

                    # 確保維度正確
                    if reconstructed_audio.ndim == 3:
                        reconstructed_audio = reconstructed_audio.squeeze(1)  # [1, T]

                    # 保存這個樣本
                    idx = saved_count

                    # 獲取 speaker ID（如果可用）
                    speaker_id = batch.get('speaker_id', [f'sample{idx}'])[sample_idx] if isinstance(batch.get('speaker_id', [f'sample{idx}']), list) else f'sample{idx}'
                    if isinstance(speaker_id, torch.Tensor):
                        speaker_id = f'speaker{speaker_id.item()}'

                    # 轉為 CPU numpy
                    noisy_np = sample_noisy_audio.cpu().numpy() if isinstance(sample_noisy_audio, torch.Tensor) else sample_noisy_audio
                    clean_np = sample_clean_audio.cpu().numpy() if isinstance(sample_clean_audio, torch.Tensor) else sample_clean_audio
                    recon_np = reconstructed_audio[0].cpu().numpy()

                    # 保存音頻文件（符合參考格式：sample_X_speaker_type.wav）
                    sf.write(sample_dir / f"sample_{idx}_{speaker_id}_noisy.wav", noisy_np, sample_rate)
                    sf.write(sample_dir / f"sample_{idx}_{speaker_id}_clean.wav", clean_np, sample_rate)
                    sf.write(sample_dir / f"sample_{idx}_{speaker_id}_pred.wav", recon_np, sample_rate)

                    # 保存頻譜圖（符合參考格式：sample_X_speaker_spectrogram.png）
                    save_spectrogram_comparison(
                        sample_noisy_audio,
                        sample_clean_audio,
                        reconstructed_audio[0],
                        sample_dir / f"sample_{idx}_{speaker_id}_spectrogram.png",
                        sample_rate=sample_rate,
                        title=f"{prefix.upper()} Sample {idx} - Epoch {epoch}"
                    )

                    saved_count += 1
                    logger.info(f"Saved {prefix} sample {idx} for epoch {epoch}")

                except Exception as e:
                    logger.warning(f"Error saving sample {sample_idx}: {str(e)}")
                    continue

    model.train()
    logger.info(f"Saved {saved_count} {prefix} samples for epoch {epoch}")


def save_training_artifacts(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    output_dir: Path,
    epoch: int,
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    num_samples: int = 3,
    sample_rate: int = 24000
):
    """
    保存所有訓練產物（每 N epoch 調用一次）

    Args:
        model: 訓練的模型
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        device: 設備
        output_dir: 輸出目錄
        epoch: 當前 epoch
        train_losses: 訓練 loss 歷史
        val_losses: 驗證 loss 歷史
        train_accs: 訓練 accuracy 歷史
        val_accs: 驗證 accuracy 歷史
        num_samples: 每個數據集保存的樣本數
        sample_rate: 採樣率
    """
    logger.info(f"Saving training artifacts for epoch {epoch}...")

    # 1. 保存 loss 圖（直接保存在根目錄，符合參考格式）
    save_loss_plot(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        output_dir / f"loss_curves_epoch_{epoch}.png",
        title=f"Training Progress - Epoch {epoch}"
    )

    # 2. 保存訓練樣本（使用 "training" 符合參考格式）
    logger.info("Saving training samples...")
    save_audio_samples(
        model, train_loader, device, output_dir, epoch,
        num_samples=num_samples, sample_rate=sample_rate, prefix="training"
    )

    # 3. 保存驗證樣本（使用 "validation" 符合參考格式）
    logger.info("Saving validation samples...")
    save_audio_samples(
        model, val_loader, device, output_dir, epoch,
        num_samples=num_samples, sample_rate=sample_rate, prefix="validation"
    )

    logger.info(f"Finished saving artifacts for epoch {epoch}")


# ============================================================================
#                          Speaker Analysis Functions
# ============================================================================

def collect_speaker_embeddings(dataset, max_samples_per_speaker=20):
    """
    收集所有 speaker embeddings（按語者分組）

    Args:
        dataset: Dataset 對象 (支持 __getitem__ 和 __len__)
        max_samples_per_speaker: 每位語者最多收集多少個樣本

    Returns:
        embeddings: (N, D) numpy array
        speaker_ids: list of speaker IDs (e.g., 'girl1', 'boy2')
    """
    from collections import defaultdict
    import numpy as np
    from tqdm import tqdm

    embeddings_by_speaker = defaultdict(list)

    # 遍歷數據集收集 speaker embeddings
    for idx in tqdm(range(len(dataset)), desc="Collecting speaker embeddings", leave=False):
        sample = dataset[idx]

        # 獲取 speaker embedding
        speaker_emb_tensor = sample.get('speaker_embedding', sample.get('speaker_embeddings'))
        speaker_emb = speaker_emb_tensor.cpu().numpy() if isinstance(speaker_emb_tensor, torch.Tensor) else speaker_emb_tensor

        # 從緩存中讀取 speaker_id
        speaker_id = sample.get('speaker_id', f'unknown_{idx}')

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
    import numpy as np

    # 合併所有 embeddings
    all_embeddings = np.concatenate([train_embeddings, val_embeddings], axis=0)
    all_speaker_ids = train_speaker_ids + val_speaker_ids

    # t-SNE 降維
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
        # 提取語者編號
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
    ax.set_title('Speaker Embedding Distribution (t-SNE)\nTrain(Blue) vs Val(Red)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.2)

    # 添加說明文字
    textstr = f'Total: {len(speaker_centers)} speakers\nTrain: {len(unique_train_speakers)}\nVal: {len(unique_val_speakers)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Speaker t-SNE visualization saved to {save_path}")


def save_speaker_analysis(train_dataset, val_dataset, output_dir):
    """
    執行完整的 speaker analysis 並保存結果

    Args:
        train_dataset: 訓練集 dataset
        val_dataset: 驗證集 dataset
        output_dir: 輸出目錄
    """
    import numpy as np
    from pathlib import Path

    speaker_dir = Path(output_dir) / 'speaker_analysis'
    speaker_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting speaker embeddings from train set...")
    train_embeddings, train_speaker_ids = collect_speaker_embeddings(train_dataset)

    logger.info("Collecting speaker embeddings from val set...")
    val_embeddings, val_speaker_ids = collect_speaker_embeddings(val_dataset)

    # 保存 embeddings 和 speaker IDs
    np.save(speaker_dir / 'train_embeddings.npy', train_embeddings)
    np.save(speaker_dir / 'val_embeddings.npy', val_embeddings)

    with open(speaker_dir / 'train_speaker_ids.txt', 'w') as f:
        f.write('\n'.join(train_speaker_ids))

    with open(speaker_dir / 'val_speaker_ids.txt', 'w') as f:
        f.write('\n'.join(val_speaker_ids))

    # 生成 t-SNE 可視化
    logger.info("Generating t-SNE visualization...")
    visualize_speaker_embeddings(
        train_embeddings, val_embeddings,
        train_speaker_ids, val_speaker_ids,
        speaker_dir / 'speaker_distribution_tsne.png'
    )

    logger.info(f"Speaker analysis completed and saved to {speaker_dir}")

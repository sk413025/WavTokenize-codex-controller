"""
Token Denoising Transformer 訓練腳本 (簡化版 - 僅 CE Loss)

簡化版本，只使用 CrossEntropy Loss

使用方式:
    python train.py \\
        --input_dirs /path/to/noisy1 /path/to/noisy2 \\
        --target_dir /path/to/clean \\
        --output_dir ./results/ce_only \\
        --num_epochs 600 \\
        --batch_size 8 \\
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
from data import AudioDataset
from model import TokenDenoisingTransformer
from loss import SimpleCELoss

# 添加必要的路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """
    解碼並儲存音頻樣本和頻譜圖
    
    Args:
        wavtokenizer: WavTokenizer 模型
        noisy_tokens: (B, T) 噪音 tokens
        pred_tokens: (B, T) 預測 tokens
        clean_tokens: (B, T) 乾淨 tokens
        epoch: 當前 epoch
        output_dir: 輸出目錄
        device: 設備
        num_samples: 保存樣本數量
    """
    samples_dir = Path(output_dir) / 'audio_samples' / f'epoch_{epoch}'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 限制樣本數量
    num_samples = min(num_samples, noisy_tokens.size(0))
    
    with torch.no_grad():
        for i in range(num_samples):
            # 完全複製 debug_single_sample.py 的解碼邏輯
            # 假設 noisy_tokens shape 可能是 (B, T) 或 (B, 1, T)
            # 統一處理：確保取出的是 2D tensor (1, T)，然後 unsqueeze(0) 變成 (1, 1, T)
            if noisy_tokens.dim() == 3:  # (B, 1, T)
                noisy_tok = noisy_tokens[i].unsqueeze(0).to(device)  # (1, T) -> (1, 1, T)
                pred_tok = pred_tokens[i].unsqueeze(0).to(device)
                clean_tok = clean_tokens[i].unsqueeze(0).to(device)
            else:  # (B, T)
                noisy_tok = noisy_tokens[i].unsqueeze(0).unsqueeze(0).to(device)  # (T) -> (1, 1, T)
                pred_tok = pred_tokens[i].unsqueeze(0).unsqueeze(0).to(device)
                clean_tok = clean_tokens[i].unsqueeze(0).unsqueeze(0).to(device)

            # 解碼為音頻 - 與 debug_single_sample.py 完全相同的流程
            noisy_features = wavtokenizer.codes_to_features(noisy_tok)
            pred_features = wavtokenizer.codes_to_features(pred_tok)
            clean_features = wavtokenizer.codes_to_features(clean_tok)

            # 與 debug_single_sample.py 相同的 squeeze 處理
            if noisy_features.dim() == 4:
                noisy_features = noisy_features.squeeze(2)
            if pred_features.dim() == 4:
                pred_features = pred_features.squeeze(2)
            if clean_features.dim() == 4:
                clean_features = clean_features.squeeze(2)

            # 確保 bandwidth_id 在正確的設備上
            bandwidth_id = torch.tensor([0], device=device)

            noisy_audio = wavtokenizer.decode(noisy_features, bandwidth_id=bandwidth_id).cpu()
            pred_audio = wavtokenizer.decode(pred_features, bandwidth_id=bandwidth_id).cpu()
            clean_audio = wavtokenizer.decode(clean_features, bandwidth_id=bandwidth_id).cpu()

            # 與 debug_single_sample.py 相同的維度檢查
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
    """
    繪製訓練和驗證的損失及準確率曲線 (簡化版 - 只有 CE Loss)

    Args:
        train_losses: 訓練損失列表 (scalar list)
        val_losses: 驗證損失列表 (scalar list)
        train_accs: 訓練準確率列表
        val_accs: 驗證準確率列表
        output_path: 圖表輸出路徑
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. CE Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CE Loss')
    axes[0].set_title('CrossEntropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Token Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Token Accuracy')
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
    """訓練一個 epoch (簡化版 - 只使用 CE Loss)"""
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (noisy_tokens, clean_tokens, content_ids) in enumerate(progress_bar):
        noisy_tokens = noisy_tokens.to(device)
        clean_tokens = clean_tokens.to(device)

        # Forward pass
        logits = model(noisy_tokens, return_logits=True)  # (B, T, 4096)

        # Compute CE loss (簡化版)
        loss = criterion(logits, clean_tokens)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item()

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
            'acc': f'{current_acc:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def validate_epoch(model, dataloader, criterion, device, epoch):
    """驗證一個 epoch (簡化版 - 只使用 CE Loss)"""
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for noisy_tokens, clean_tokens, content_ids in tqdm(dataloader, desc="Validation"):
            noisy_tokens = noisy_tokens.to(device)
            clean_tokens = clean_tokens.to(device)

            # Forward pass
            logits = model(noisy_tokens, return_logits=True)  # (B, T, 4096)

            # Compute CE loss (簡化版)
            loss = criterion(logits, clean_tokens)

            # Statistics
            total_loss += loss.item()

            # Token accuracy
            B, T, _ = logits.shape
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
    parser = argparse.ArgumentParser(description='Token Denoising Transformer Training (CE Loss Only)')

    # 數據參數
    parser.add_argument('--input_dirs', nargs='+', required=True, help='含噪音輸入目錄')
    parser.add_argument('--target_dir', required=True, help='乾淨目標目錄')
    parser.add_argument('--output_dir', default='./results/ce_only', help='輸出目錄')
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
    
    # WavTokenizer 參數
    parser.add_argument('--wavtokenizer_config', 
                       default='/home/sbplab/ruizi/WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                       help='WavTokenizer 配置文件')
    parser.add_argument('--wavtokenizer_checkpoint',
                       default='/home/sbplab/ruizi/WavTokenizer/results/smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_epoch_1200.pth',
                       help='WavTokenizer checkpoint')
    
    args = parser.parse_args()
    
    # 創建輸出目錄和所有必要的子目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提前創建所有需要的子目錄
    (output_dir / 'audio_samples').mkdir(parents=True, exist_ok=True)
    
    # 設置 logger
    logger = setup_logger(output_dir)
    logger.info("=" * 80)
    logger.info("Token Denoising Transformer 訓練 (簡化版 - 僅 CE Loss)")
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
    
    # 提取 Codebook (第一層量化器)
    # 正確路徑: feature_extractor.encodec.quantizer.vq.layers[0].codebook
    codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
    logger.info(f"Codebook 形狀: {codebook.shape}")  # (4096, 512)
    
    # 創建數據集
    logger.info("創建數據集...")
    logger.info(f"輸入目錄: {args.input_dirs}")
    logger.info(f"目標目錄: {args.target_dir}")
    logger.info(f"每個語者句子數: {args.max_sentences_per_speaker if args.max_sentences_per_speaker else '全部'}")
    
    # Token collate function (修改為返回 content_ids)
    def token_collate_fn(batch):
        """Collate function with content IDs"""
        noisy_tokens_list = []
        clean_tokens_list = []
        content_ids_list = []
        
        for noisy_audio, clean_audio, content_id in batch:
            noisy_audio = noisy_audio.to(device).unsqueeze(0)
            clean_audio = clean_audio.to(device).unsqueeze(0)
            
            # 編碼為 tokens
            with torch.no_grad():
                _, noisy_tokens = wavtokenizer.encode_infer(
                    noisy_audio, 
                    bandwidth_id=torch.tensor([0], device=device)
                )
                _, clean_tokens = wavtokenizer.encode_infer(
                    clean_audio, 
                    bandwidth_id=torch.tensor([0], device=device)
                )
            
            noisy_tokens_list.append(noisy_tokens[0])  # [1, seq_len]
            clean_tokens_list.append(clean_tokens[0])
            content_ids_list.append(content_id)
        
        # 找最大長度
        max_len = max(
            max(t.shape[1] for t in noisy_tokens_list),
            max(t.shape[1] for t in clean_tokens_list)
        )
        
        # Pad tokens
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
        
        # 將 content_id 轉換為整數（如果是字串）
        numeric_ids = []
        for cid in content_ids_list:
            if isinstance(cid, str):
                # 提取數字部分，如 "LDV_001" -> 1
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
    
    # ============================================================
    # 詳細的數據集統計信息
    # ============================================================
    logger.info("=" * 80)
    logger.info("數據集統計信息")
    logger.info("=" * 80)
    
    # 統計材質
    materials = set()
    for input_dir in args.input_dirs:
        material = os.path.basename(input_dir)
        materials.add(material)
    logger.info(f"使用材質: {', '.join(sorted(materials))}")
    
    # 統計語者和句子
    speaker_stats = {}
    content_ids = set()
    content_id_counts = {}  # 統計每個 content_id 的出現次數
    
    for pair in audio_dataset.paired_files:
        filename = pair['input']
        # 檔名格式可能有兩種:
        # 1. noisy: nor_boy10_box_LDV_001.wav (5個部分)
        # 2. clean: nor_boy10_clean_001.wav (4個部分)
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 5:
            # noisy 格式: nor_boy10_box_LDV_001.wav
            speaker = parts[1]  # boy10, girl9, etc.
            content_id = parts[4].replace('.wav', '')  # 001
        elif len(parts) >= 4:
            # clean 格式: nor_boy10_clean_001.wav
            speaker = parts[1]  # boy10, girl9, etc.
            content_id = parts[3].replace('.wav', '')  # 001
        else:
            speaker = 'unknown'
            content_id = 'unknown'
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = 0
        speaker_stats[speaker] += 1
        content_ids.add(content_id)
        
        # 統計每個 content_id 的出現次數
        if content_id not in content_id_counts:
            content_id_counts[content_id] = 0
        content_id_counts[content_id] += 1
    
    logger.info(f"總數據集大小: {len(audio_dataset)} 個音頻對")
    logger.info(f"語者數量: {len(speaker_stats)}")
    logger.info(f"不同句子 (content_id): {len(content_ids)}")
    
    # 新增：顯示 content_id 分布統計
    logger.info("")
    logger.info("Content ID 分布統計:")
    logger.info(f"  總共有 {len(content_ids)} 個不同的句子")
    
    # 按出現次數排序
    sorted_content_ids = sorted(content_id_counts.items(), key=lambda x: x[1], reverse=True)
    logger.info("  每個句子的重複次數 (前 20 個):")
    for content_id, count in sorted_content_ids[:20]:
        logger.info(f"    - {content_id}: {count} 次")
    
    # 統計重複次數的分布
    repeat_counts = {}
    for content_id, count in content_id_counts.items():
        if count not in repeat_counts:
            repeat_counts[count] = 0
        repeat_counts[count] += 1
    
    logger.info("")
    logger.info("  重複次數分布:")
    for repeat_count in sorted(repeat_counts.keys()):
        num_sentences = repeat_counts[repeat_count]
        logger.info(f"    - 重複 {repeat_count} 次: {num_sentences} 個句子")
    
    # 計算平均重複次數
    total_repeats = sum(content_id_counts.values())
    avg_repeats = total_repeats / len(content_ids) if content_ids else 0
    logger.info(f"  平均每個句子重複: {avg_repeats:.1f} 次")
    
    # 重要警告：如果平均重複次數太低，內容一致性損失效果會不好
    if avg_repeats < 2.0:
        logger.warning("")
        logger.warning("  ⚠️ 警告：平均重複次數 < 2.0")
        logger.warning("  ⚠️ 這意味著大部分句子只出現 1 次，無法計算內容一致性損失")
        logger.warning("  ⚠️ Content Consistency Loss 可能效果不佳")
        logger.warning("")
    else:
        logger.info(f"  ✅ 平均重複次數 = {avg_repeats:.1f}，適合計算內容一致性損失")
    logger.info("")
    
    # 顯示每個語者的句子數
    logger.info("各語者句子數:")
    for speaker in sorted(speaker_stats.keys()):
        count = speaker_stats[speaker]
        logger.info(f"  - {speaker}: {count} 句")
    
    # 計算統計信息
    sentences_per_speaker = list(speaker_stats.values())
    avg_sentences = sum(sentences_per_speaker) / len(sentences_per_speaker) if sentences_per_speaker else 0
    logger.info(f"平均每位語者句子數: {avg_sentences:.1f}")
    logger.info(f"每位語者句子數範圍: {min(sentences_per_speaker) if sentences_per_speaker else 0} - {max(sentences_per_speaker) if sentences_per_speaker else 0}")
    
    logger.info("=" * 80)
    logger.info("數據集分割 (按語者)")
    logger.info("=" * 80)
    
    # 按語者分割數據集（參考 train_token_denoising.py）
    # 驗證集語者：girl9, girl10, boy7, boy8
    # 訓練集語者：其他所有語者
    val_speakers = ['girl9', 'girl10', 'boy7', 'boy8']
    train_speakers = ['boy1', 'boy3', 'boy4', 'boy5', 'boy6', 'boy9', 'boy10', 
                     'girl2', 'girl3', 'girl4', 'girl6', 'girl7', 'girl8', 'girl11']
    
    logger.info(f"驗證集語者: {val_speakers}")
    logger.info(f"訓練集語者: {train_speakers}")
    
    train_indices = []
    val_indices = []
    
    for idx in range(len(audio_dataset)):
        # AudioDataset 使用 paired_files 屬性
        filename = audio_dataset.paired_files[idx]['input']
        # 文件名格式: nor_boy10_box_LDV_001.wav
        # 提取語者名稱 (第二個部分)
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 2:
            speaker = parts[1]  # boy10, girl9, etc.
        else:
            speaker = parts[0]  # fallback
        
        if speaker in val_speakers:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    
    train_audio_dataset = torch.utils.data.Subset(audio_dataset, train_indices)
    val_audio_dataset = torch.utils.data.Subset(audio_dataset, val_indices)
    
    # 統計訓練集和驗證集中的語者分布
    train_speaker_counts = {}
    val_speaker_counts = {}
    
    for idx in train_indices:
        pair = audio_dataset.paired_files[idx]
        filename = pair['input']
        parts = os.path.basename(filename).split('_')
        speaker = parts[1] if len(parts) >= 2 else 'unknown'
        train_speaker_counts[speaker] = train_speaker_counts.get(speaker, 0) + 1
    
    for idx in val_indices:
        pair = audio_dataset.paired_files[idx]
        filename = pair['input']
        parts = os.path.basename(filename).split('_')
        speaker = parts[1] if len(parts) >= 2 else 'unknown'
        val_speaker_counts[speaker] = val_speaker_counts.get(speaker, 0) + 1
    
    logger.info(f"訓練集大小: {len(train_audio_dataset)} ({len(train_audio_dataset)/len(audio_dataset)*100:.1f}%)")
    logger.info(f"驗證集大小: {len(val_audio_dataset)} ({len(val_audio_dataset)/len(audio_dataset)*100:.1f}%)")
    
    logger.info("訓練集語者分布:")
    for speaker in sorted(train_speaker_counts.keys()):
        count = train_speaker_counts[speaker]
        logger.info(f"  - {speaker}: {count} 句")
    
    logger.info("驗證集語者分布:")
    for speaker in sorted(val_speaker_counts.keys()):
        count = val_speaker_counts[speaker]
        logger.info(f"  - {speaker}: {count} 句")
    
    logger.info("=" * 80)
    
    # DataLoader - 使用標準批次採樣
    logger.info("使用標準批次採樣")
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
    
    # 確認 Codebook 凍結
    assert not model.codebook.requires_grad, "Codebook 必須凍結！"
    
    # 計算參數量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(b.numel() for b in model.buffers())
    total_params = trainable_params + frozen_params
    
    logger.info("模型架構:")
    logger.info(f"  - d_model: {args.d_model}")
    logger.info(f"  - nhead: {args.nhead}")
    logger.info(f"  - num_layers: {args.num_layers}")
    logger.info(f"  - dim_feedforward: {args.dim_feedforward}")
    logger.info(f"  - dropout: {args.dropout}")
    logger.info("")
    logger.info(f"模型總參數數量: {total_params:,}")
    logger.info(f"  - 可訓練參數: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"  - 凍結參數 (Codebook): {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    logger.info(f"  - Codebook 形狀: {model.codebook.shape}")
    logger.info(f"  - Codebook 凍結: ✓ {not model.codebook.requires_grad}")
    logger.info("=" * 80)
    
    # 損失函數配置
    logger.info("=" * 80)
    logger.info("損失函數配置 (簡化版)")
    logger.info("=" * 80)
    logger.info("損失組成:")
    logger.info("  - CrossEntropy Loss (Token 準確度)")
    logger.info("=" * 80)

    # 損失函數 (簡化版 - 只使用 CE Loss)
    criterion = SimpleCELoss(device=device)
    
    # 優化器
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # 學習率調度器 - 使用 ReduceLROnPlateau 以更好地處理過擬合
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True, min_lr=1e-6
    )
    
    # 訓練歷史記錄（用於繪圖）
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
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

        # 更新學習率 - 根據驗證損失調整
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        
        # 保存 checkpoint (每 10 epochs)
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
        
        # 每 50 epochs 繪製損失曲線
        if epoch % 50 == 0 and epoch > 0:
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
                logger.info(f"  ✓ 已保存損失曲線到: {plot_path.name}")
            except Exception as e:
                logger.error(f"  繪製損失曲線時出錯: {e}")

        # 每 100 epochs 保存音頻樣本和頻譜圖
        if epoch % 100 == 0 or epoch == args.num_epochs - 1:
            logger.info(f"  保存音頻樣本（訓練集 - inside test）...")
            try:
                # ⭐ 關鍵修正：使用 DataLoader 中的 tokens（訓練時模型看到的格式）
                # 而不是重新 encode 音檔，這樣才能保證訓練和預測的數據一致性
                model.eval()

                # 從訓練集 loader 取一個 batch（檢查 overfitting）
                train_batch = next(iter(train_loader))
                noisy_tokens_batch, clean_tokens_batch, content_ids = train_batch

                # 取前 3 個樣本
                num_save = min(3, noisy_tokens_batch.shape[0])

                noisy_tokens_list = []
                clean_tokens_list = []
                pred_tokens_list = []

                for i in range(num_save):
                    # 使用 DataLoader 產生的 tokens（已經過 collate_fn 處理）
                    noisy_tok = noisy_tokens_batch[i:i+1]  # [1, T_padded]
                    clean_tok = clean_tokens_batch[i:i+1]  # [1, T_padded]

                    # 預測（使用與訓練時相同格式的 tokens）
                    with torch.no_grad():
                        pred_logits = model(noisy_tok, return_logits=True)  # [1, T, vocab]
                        pred_tok = pred_logits.argmax(dim=-1)  # [1, T]

                    # 儲存 tokens
                    noisy_tokens_list.append(noisy_tok)
                    clean_tokens_list.append(clean_tok)
                    pred_tokens_list.append(pred_tok)

                # 直接傳入 list，逐個保存（避免 padding）
                samples_dir = Path(output_dir) / 'audio_samples' / f'epoch_{epoch}'
                samples_dir.mkdir(parents=True, exist_ok=True)

                for idx in range(num_save):
                    noisy_tok = noisy_tokens_list[idx]  # [1, T]
                    clean_tok = clean_tokens_list[idx]  # [1, T]
                    pred_tok = pred_tokens_list[idx]    # [1, T]

                    # 解碼為音頻（⭐ 關鍵：與 debug_single_sample.py 完全相同的流程）
                    with torch.no_grad():
                        # ⭐ debug: [B,T] -> tokens[i] -> [T] -> unsqueeze(0) -> [1,T] -> codes_to_features
                        # 這裡: [1,T] -> squeeze(0) -> [T] -> unsqueeze(0) -> [1,T] -> codes_to_features
                        noisy_tok_2d = noisy_tok.squeeze(0).unsqueeze(0).to(device)  # [1,T] -> [T] -> [1,T]
                        pred_tok_2d = pred_tok.squeeze(0).unsqueeze(0).to(device)
                        clean_tok_2d = clean_tok.squeeze(0).unsqueeze(0).to(device)

                        noisy_features = wavtokenizer.codes_to_features(noisy_tok_2d)
                        pred_features = wavtokenizer.codes_to_features(pred_tok_2d)
                        clean_features = wavtokenizer.codes_to_features(clean_tok_2d)

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

                    # 保存音頻
                    torchaudio.save(str(samples_dir / f'sample_{idx}_noisy.wav'), noisy_audio, 24000)
                    torchaudio.save(str(samples_dir / f'sample_{idx}_predicted.wav'), pred_audio, 24000)
                    torchaudio.save(str(samples_dir / f'sample_{idx}_clean.wav'), clean_audio, 24000)

                    # 保存頻譜圖
                    plot_spectrograms(
                        noisy_audio.squeeze(0).numpy(),
                        pred_audio.squeeze(0).numpy(),
                        clean_audio.squeeze(0).numpy(),
                        str(samples_dir / f'sample_{idx}_spectrogram.png')
                    )
                logger.info(f"  ✓ 已保存音頻樣本到: audio_samples/epoch_{epoch}/")
                model.train()
            except Exception as e:
                logger.error(f"  保存音頻樣本時出錯: {e}")
    
    logger.info("訓練完成！")
    logger.info(f"最佳驗證損失: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

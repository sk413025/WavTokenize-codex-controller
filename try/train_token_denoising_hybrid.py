"""
Token Denoising Transformer 訓練腳本 (混合損失版本)

結合：
1. CrossEntropy Loss (token 準確度)
2. Content Consistency Loss (相同內容應相似)
3. Embedding L2 Loss (embedding 空間接近)

使用方式:
    python train_token_denoising_hybrid.py \\
        --input_dirs /path/to/noisy1 /path/to/noisy2 \\
        --target_dir /path/to/clean \\
        --output_dir ./results/hybrid_loss \\
        --num_epochs 600 \\
        --batch_size 8 \\
        --max_sentences_per_speaker 288 \\
        --ce_weight 1.0 \\
        --content_weight 0.5 \\
        --embed_weight 0.3
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
import random
from collections import defaultdict

# 添加必要的路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from ttdata import AudioDataset
from token_denoising_transformer import TokenDenoisingTransformer
from decoder.pretrained import WavTokenizer
from discrete_hybrid_loss import DiscreteHybridLoss
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
            # 擴展 tokens 到 (1, B, T) 格式 (單層量化器)
            noisy_tok = noisy_tokens[i:i+1].unsqueeze(0)
            pred_tok = pred_tokens[i:i+1].unsqueeze(0)
            clean_tok = clean_tokens[i:i+1].unsqueeze(0)
            
            # 解碼為音頻 - 修復維度問題
            noisy_features = wavtokenizer.codes_to_features(noisy_tok)
            pred_features = wavtokenizer.codes_to_features(pred_tok)
            clean_features = wavtokenizer.codes_to_features(clean_tok)
            
            # 檢查並修正維度：codes_to_features 可能返回 4D [1, T, 1, D]
            # 需要轉換為 3D [1, T, D] 供 decode 使用
            if noisy_features.dim() == 4:
                noisy_features = noisy_features.squeeze(2)  # [1, T, 1, D] -> [1, T, D]
            if pred_features.dim() == 4:
                pred_features = pred_features.squeeze(2)
            if clean_features.dim() == 4:
                clean_features = clean_features.squeeze(2)
            
            noisy_audio = wavtokenizer.decode(noisy_features, bandwidth_id=torch.tensor([0]))
            pred_audio = wavtokenizer.decode(pred_features, bandwidth_id=torch.tensor([0]))
            clean_audio = wavtokenizer.decode(clean_features, bandwidth_id=torch.tensor([0]))
            
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
    繪製訓練和驗證的損失及準確率曲線
    
    Args:
        train_losses: 訓練損失列表 (每個epoch一個dict)
        val_losses: 驗證損失列表 (每個epoch一個dict)
        train_accs: 訓練準確率列表
        val_accs: 驗證準確率列表
        output_path: 圖表輸出路徑
    """
    epochs = list(range(1, len(train_losses) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 總損失
    axes[0, 0].plot(epochs, [m['total_loss'] for m in train_losses], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, [m['total_loss'] for m in val_losses], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 準確率
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Token Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. CE Loss
    axes[1, 0].plot(epochs, [m['ce_loss'] for m in train_losses], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, [m['ce_loss'] for m in val_losses], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('CE Loss')
    axes[1, 0].set_title('CrossEntropy Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Content + Embed Loss
    axes[1, 1].plot(epochs, [m['content_loss'] for m in train_losses], 'b-', label='Train Content', linewidth=2)
    axes[1, 1].plot(epochs, [m['content_loss'] for m in val_losses], 'r-', label='Val Content', linewidth=2)
    axes[1, 1].plot(epochs, [m['embed_loss'] for m in train_losses], 'g--', label='Train Embed', linewidth=2)
    axes[1, 1].plot(epochs, [m['embed_loss'] for m in val_losses], 'orange', linestyle='--', label='Val Embed', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Content & Embedding Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {output_path}")


class ContentAwareBatchSampler:
    """
    內容感知批次採樣器
    
    確保每個批次都包含足夠的相同content_id樣本以計算內容一致性損失
    
    參考自 ttt2.py 的實現
    
    Args:
        dataset: 包含content_id的音訊數據集
        batch_size: 每個批次的大小
        content_ratio: 每個批次中相同content_id樣本的比例 (0.0-1.0)
        min_content_samples: 每個批次中相同內容ID的最小樣本數
        shuffle: 是否隨機打亂批次順序
        drop_last: 是否丟棄最後一個不完整的批次
    """
    def __init__(self, dataset, batch_size=8, content_ratio=0.5, min_content_samples=3, 
                 shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.content_ratio = content_ratio
        self.min_content_samples = min_content_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 對所有樣本按content_id分組
        self.content_groups = self._group_by_content_id()
        
        # 檢查每個內容ID的樣本數
        self._validate_and_report_groups()
        
        # 創建批次索引
        self.batch_indices = self._create_batch_indices()
        
    def _group_by_content_id(self):
        """將數據集樣本按content_id分組"""
        content_groups = defaultdict(list)
        
        # 檢查是否為Subset類型
        if hasattr(self.dataset, 'dataset') and hasattr(self.dataset, 'indices'):
            # 處理PyTorch的Subset類型
            original_dataset = self.dataset.dataset
            subset_indices = self.dataset.indices
            
            # 根據Subset索引更新content_groups
            for i, idx in enumerate(subset_indices):
                if hasattr(original_dataset, 'paired_files'):
                    # AudioDataset情況：從檔名提取content_id
                    filename = original_dataset.paired_files[idx]['input']
                    parts = os.path.basename(filename).split('_')
                    if len(parts) >= 5:
                        # 檔名格式: nor_boy10_box_LDV_001.wav
                        # content_id 是句子編號 (001, 002, ...)
                        content_id = parts[4].replace('.wav', '')
                    else:
                        content_id = f"unknown_{idx}"
                    content_groups[content_id].append(i)
        else:
            # 直接處理普通數據集
            if hasattr(self.dataset, 'paired_files'):
                # AudioDataset情況
                for i, pair in enumerate(self.dataset.paired_files):
                    filename = pair['input']
                    parts = os.path.basename(filename).split('_')
                    if len(parts) >= 5:
                        # 檔名格式: nor_boy10_box_LDV_001.wav
                        # content_id 是句子編號 (001, 002, ...)
                        content_id = parts[4].replace('.wav', '')
                    else:
                        content_id = f"unknown_{i}"
                    content_groups[content_id].append(i)
            else:
                # 不明確的數據集類型，使用索引作為內容ID
                for i in range(len(self.dataset)):
                    content_groups[f"idx_{i}"].append(i)
                print("警告: 無法確定數據集類型，使用索引作為內容ID")
        
        return content_groups
        
    def _validate_and_report_groups(self):
        """驗證並報告分組情況"""
        if not self.content_groups:
            print("警告: 沒有找到任何內容ID分組")
            return
            
        # 統計每個內容ID的樣本數
        id_counts = {cid: len(indices) for cid, indices in self.content_groups.items()}
        
        # 找出有足夠樣本數的內容ID
        valid_ids = {cid: count for cid, count in id_counts.items() 
                    if count >= self.min_content_samples}
        
        print(f"\n內容ID分組統計:")
        print(f"總內容ID數量: {len(self.content_groups)}")
        print(f"有效內容ID數量 (樣本數 >= {self.min_content_samples}): {len(valid_ids)}")
        
        if len(valid_ids) < len(self.content_groups):
            print(f"警告: {len(self.content_groups) - len(valid_ids)} 個內容ID樣本數不足 {self.min_content_samples}")
        
        # 統計分布情況
        sample_counts = list(id_counts.values())
        if sample_counts:
            print(f"每個內容ID的樣本數: 最小={min(sample_counts)}, "
                  f"最大={max(sample_counts)}, 平均={sum(sample_counts)/len(sample_counts):.1f}")
        
    def _create_batch_indices(self):
        """創建批次索引列表"""
        batch_indices = []
        
        # 獲取所有可用的索引集合
        available_indices = set(range(len(self.dataset)))
        
        # 內容ID組中至少有min_content_samples個樣本的ID列表
        valid_content_ids = [
            cid for cid, indices in self.content_groups.items()
            if len(indices) >= self.min_content_samples
        ]
        
        if not valid_content_ids:
            print("警告: 沒有足夠的內容ID組，使用普通批次劃分")
            # 回退到普通批次劃分
            indices_list = list(available_indices)
            if self.shuffle:
                random.shuffle(indices_list)
            
            for i in range(0, len(indices_list), self.batch_size):
                batch = indices_list[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batch_indices.append(batch)
            return batch_indices
        
        # 根據內容感知策略創建批次
        while available_indices and valid_content_ids:
            batch = []
            
            # 步驟1: 隨機選擇一個有效的內容ID
            selected_cid = random.choice(valid_content_ids)
            
            # 該內容ID組的可用索引
            cid_indices = [idx for idx in self.content_groups[selected_cid] 
                          if idx in available_indices]
            
            # 如果此內容ID沒有足夠的樣本，則從有效ID列表中移除
            if len(cid_indices) < self.min_content_samples:
                valid_content_ids.remove(selected_cid)
                continue
            
            # 步驟2: 確定要從該內容ID組中選取的樣本數
            content_samples = max(
                self.min_content_samples, 
                min(len(cid_indices), int(self.batch_size * self.content_ratio))
            )
            
            # 選擇內容ID樣本
            selected_indices = random.sample(cid_indices, content_samples)
            batch.extend(selected_indices)
            
            # 從可用索引中移除已選擇的索引
            for idx in selected_indices:
                available_indices.remove(idx)
            
            # 步驟3: 用其他內容ID的樣本填滿批次
            remaining_slots = self.batch_size - len(batch)
            
            if remaining_slots > 0 and available_indices:
                # 優先選擇不同內容ID的樣本
                other_indices = [idx for idx in available_indices 
                               if all(idx not in self.content_groups[cid] 
                                      for cid in [selected_cid])]
                
                # 如果其他內容ID的樣本不足，就使用任何可用樣本
                if len(other_indices) < remaining_slots:
                    other_indices = list(available_indices)
                
                # 隨機選擇剩餘樣本
                fill_indices = random.sample(
                    other_indices, 
                    min(remaining_slots, len(other_indices))
                )
                
                batch.extend(fill_indices)
                
                # 從可用索引中移除已選擇的索引
                for idx in fill_indices:
                    available_indices.remove(idx)
            
            batch_indices.append(batch)
            
            # 更新有效內容ID列表
            valid_content_ids = [
                cid for cid in valid_content_ids
                if sum(1 for idx in self.content_groups[cid] if idx in available_indices) >= self.min_content_samples
            ]
        
        # 如果還有剩餘索引，創建額外的批次
        if available_indices and not self.drop_last:
            remaining = list(available_indices)
            if self.shuffle:
                random.shuffle(remaining)
            
            for i in range(0, len(remaining), self.batch_size):
                batch = remaining[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batch_indices.append(batch)
        
        # 如果需要，打亂批次順序
        if self.shuffle:
            random.shuffle(batch_indices)
        
        # 分析批次組成，用於調試
        self._analyze_batches(batch_indices)
        
        return batch_indices
        
    def _analyze_batches(self, batch_indices):
        """分析批次組成，顯示統計信息"""
        if not batch_indices:
            return
            
        # 統計有效批次數量 (包含足夠相同內容ID樣本的批次)
        valid_batches = 0
        content_counts_per_batch = []
        
        for batch in batch_indices:
            # 計算每個批次中每個內容ID的樣本數
            batch_content_counts = defaultdict(int)
            
            for idx in batch:
                # 找出該索引對應的內容ID
                for cid, indices in self.content_groups.items():
                    if idx in indices:
                        batch_content_counts[cid] += 1
                        break
            
            # 找出該批次中樣本數最多的內容ID
            if batch_content_counts:
                max_count = max(batch_content_counts.values())
                content_counts_per_batch.append(max_count)
                
                if max_count >= self.min_content_samples:
                    valid_batches += 1
        
        # 打印分析結果
        print(f"\n批次分析:")
        print(f"總批次數: {len(batch_indices)}")
        print(f"有效批次數 (至少含{self.min_content_samples}個相同內容ID): {valid_batches} ({valid_batches/len(batch_indices)*100:.1f}%)")
        
        if content_counts_per_batch:
            avg_content_count = sum(content_counts_per_batch) / len(content_counts_per_batch)
            print(f"每批次中相同內容ID的平均最大樣本數: {avg_content_count:.2f}")
    
    def __iter__(self):
        # 返回批次索引的迭代器
        return iter(self.batch_indices)
    
    def __len__(self):
        # 返回批次數量
        return len(self.batch_indices)


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


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    """訓練一個 epoch (使用混合損失)"""
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_content_loss = 0.0
    total_embed_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    # 統計 content_id 分布
    batch_content_stats = {
        'total_batches': 0,
        'batches_with_repeats': 0,
        'total_unique_ids': 0,
        'total_samples': 0,
        'max_repeat_in_batch': 0
    }
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (noisy_tokens, clean_tokens, content_ids) in enumerate(progress_bar):
        noisy_tokens = noisy_tokens.to(device)
        clean_tokens = clean_tokens.to(device)
        content_ids = content_ids.to(device)
        
        # 統計每個 batch 的 content_ids 分布
        unique_ids, counts = torch.unique(content_ids, return_counts=True)
        batch_size = len(content_ids)
        num_unique = len(unique_ids)
        max_repeat = counts.max().item()
        
        batch_content_stats['total_batches'] += 1
        batch_content_stats['total_samples'] += batch_size
        batch_content_stats['total_unique_ids'] += num_unique
        if num_unique < batch_size:
            batch_content_stats['batches_with_repeats'] += 1
        batch_content_stats['max_repeat_in_batch'] = max(
            batch_content_stats['max_repeat_in_batch'], 
            max_repeat
        )
        
        # 首個 batch 或前 3 個 batch 詳細檢查
        if batch_idx < 3 and epoch == 1:
            logger.info(f"")
            logger.info(f"Batch {batch_idx} 的 content_ids 分布:")
            logger.info(f"  Batch size: {batch_size}")
            logger.info(f"  唯一 content_id 數量: {num_unique}")
            logger.info(f"  Content IDs: {content_ids.cpu().tolist()}")
            logger.info(f"  唯一 IDs 及其計數: {dict(zip(unique_ids.cpu().tolist(), counts.cpu().tolist()))}")
            if num_unique < batch_size:
                logger.info(f"  ✅ Batch 中有重複的 content_id (最多重複 {max_repeat} 次)")
                logger.info(f"  ✅ 可以計算內容一致性損失")
            else:
                logger.warning(f"  ⚠️ Batch 中所有 content_id 都不同")
                logger.warning(f"  ⚠️ 此 batch 的內容一致性損失將為 0")
        
        # Forward pass
        logits = model(noisy_tokens, return_logits=True)  # (B, T, 4096)
        
        # Compute hybrid loss
        loss_dict = criterion(
            pred_logits=logits,
            target_tokens=clean_tokens,
            noisy_tokens=noisy_tokens,
            content_ids=content_ids,
            current_epoch=epoch,
            total_epochs=total_epochs
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_ce_loss += loss_dict['ce_loss']
        total_content_loss += loss_dict['content_loss']
        total_embed_loss += loss_dict['embed_loss']
        
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
            'ce': f'{loss_dict["ce_loss"]:.3f}',
            'cont': f'{loss_dict["content_loss"]:.3f}',
            'emb': f'{loss_dict["embed_loss"]:.3f}',
            'acc': f'{current_acc:.2f}%',
            'cw': f'{loss_dict["content_weight"]:.3f}'
        })
    
    # Epoch 結束後打印統計
    avg_unique_per_batch = batch_content_stats['total_unique_ids'] / batch_content_stats['total_batches']
    repeat_ratio = batch_content_stats['batches_with_repeats'] / batch_content_stats['total_batches'] * 100
    
    logger.info("")
    logger.info(f"Epoch {epoch} Content ID 統計:")
    logger.info(f"  總 batches: {batch_content_stats['total_batches']}")
    logger.info(f"  有重複 content_id 的 batches: {batch_content_stats['batches_with_repeats']} ({repeat_ratio:.1f}%)")
    logger.info(f"  平均每個 batch 的唯一 ID 數: {avg_unique_per_batch:.1f}")
    logger.info(f"  單個 batch 中最大重複次數: {batch_content_stats['max_repeat_in_batch']}")
    
    if repeat_ratio < 50:
        logger.warning(f"  ⚠️ 只有 {repeat_ratio:.1f}% 的 batches 有重複 content_id")
        logger.warning(f"  ⚠️ Content Consistency Loss 可能效果有限")
    else:
        logger.info(f"  ✅ {repeat_ratio:.1f}% 的 batches 有重複 content_id，適合訓練")
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_content_loss = total_content_loss / len(dataloader)
    avg_embed_loss = total_embed_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100
    
    return {
        'total_loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'content_loss': avg_content_loss,
        'embed_loss': avg_embed_loss,
        'accuracy': accuracy
    }


def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs):
    """驗證一個 epoch"""
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_content_loss = 0.0
    total_embed_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for noisy_tokens, clean_tokens, content_ids in tqdm(dataloader, desc="Validation"):
            noisy_tokens = noisy_tokens.to(device)
            clean_tokens = clean_tokens.to(device)
            content_ids = content_ids.to(device)
            
            # Forward pass
            logits = model(noisy_tokens, return_logits=True)
            
            # Compute loss
            loss_dict = criterion(
                pred_logits=logits,
                target_tokens=clean_tokens,
                noisy_tokens=noisy_tokens,
                content_ids=content_ids,
                current_epoch=epoch,
                total_epochs=total_epochs
            )
            
            # Statistics
            total_loss += loss_dict['total_loss'].item()
            total_ce_loss += loss_dict['ce_loss']
            total_content_loss += loss_dict['content_loss']
            total_embed_loss += loss_dict['embed_loss']
            
            # Token accuracy
            B, T, _ = logits.shape
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == clean_tokens).sum().item()
            total_correct += correct
            total_tokens += B * T
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_content_loss = total_content_loss / len(dataloader)
    avg_embed_loss = total_embed_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100
    
    return {
        'total_loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'content_loss': avg_content_loss,
        'embed_loss': avg_embed_loss,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='Token Denoising Transformer Training (Hybrid Loss)')
    
    # 數據參數
    parser.add_argument('--input_dirs', nargs='+', required=True, help='含噪音輸入目錄')
    parser.add_argument('--target_dir', required=True, help='乾淨目標目錄')
    parser.add_argument('--output_dir', default='./results/hybrid_loss', help='輸出目錄')
    parser.add_argument('--max_sentences_per_speaker', type=int, default=None,
                       help='每個語者最多使用的句子數 (None=全部)')
    
    # 模型參數
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 維度')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 層數 (降低以防止過擬合)')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN 維度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (提高以防止過擬合)')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=600, help='訓練 epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (提高以防止過擬合)')
    
    # 混合損失權重
    parser.add_argument('--ce_weight', type=float, default=1.0, 
                       help='CrossEntropy 權重')
    parser.add_argument('--content_weight', type=float, default=0.5, 
                       help='Content Consistency 最大權重 (會動態衰減)')
    parser.add_argument('--embed_weight', type=float, default=0.3, 
                       help='Embedding L2 權重')
    parser.add_argument('--warmup_epochs', type=int, default=50,
                       help='Content loss warmup epochs')
    
    # 內容感知採樣參數
    parser.add_argument('--use_content_aware', action='store_true',
                       help='使用內容感知批次採樣器')
    parser.add_argument('--content_ratio', type=float, default=0.5,
                       help='每個 batch 中相同 content_id 樣本的比例 (0.0-1.0)')
    parser.add_argument('--min_content_samples', type=int, default=3,
                       help='每個 batch 中相同 content_id 的最小樣本數')
    
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
    logger.info("Token Denoising Transformer 訓練 (混合損失版本)")
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
        # 檔名格式: nor_boy10_box_LDV_001.wav
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 5:
            speaker = parts[1]  # boy10, girl9, etc.
            # content_id 應該是句子編號 (001, 002, ...)，而非 LDV
            content_id = parts[4].replace('.wav', '')  # 001
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
    
    # DataLoader - 使用內容感知批次採樣器（如果啟用）
    if args.use_content_aware:
        logger.info("=" * 80)
        logger.info("使用內容感知批次採樣器")
        logger.info("=" * 80)
        logger.info(f"配置:")
        logger.info(f"  - content_ratio: {args.content_ratio} (每個batch中相同內容的比例)")
        logger.info(f"  - min_content_samples: {args.min_content_samples} (最少相同內容樣本數)")
        logger.info("")
        
        # 創建訓練集的批次採樣器
        logger.info("正在為訓練集創建內容感知批次索引...")
        train_batch_sampler = ContentAwareBatchSampler(
            train_audio_dataset,
            batch_size=args.batch_size,
            content_ratio=args.content_ratio,
            min_content_samples=args.min_content_samples,
            shuffle=True,
            drop_last=False
        )
        
        train_loader = DataLoader(
            train_audio_dataset,
            batch_sampler=train_batch_sampler,  # 使用 batch_sampler 替代 batch_size 和 shuffle
            num_workers=0,
            collate_fn=token_collate_fn
        )
        logger.info("✓ 訓練集 DataLoader 創建完成")
        
        # 創建驗證集的批次採樣器（使用更寬鬆的要求）
        logger.info("")
        logger.info("正在為驗證集創建內容感知批次索引...")
        val_batch_sampler = ContentAwareBatchSampler(
            val_audio_dataset,
            batch_size=args.batch_size,
            content_ratio=args.content_ratio,
            min_content_samples=2,  # 驗證集使用更寬鬆的要求
            shuffle=False,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_audio_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=0,
            collate_fn=token_collate_fn
        )
        logger.info("✓ 驗證集 DataLoader 創建完成")
        logger.info("=" * 80)
    else:
        # 標準 DataLoader
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
    
    # 創建混合損失函數
    logger.info("=" * 80)
    logger.info("混合損失函數配置")
    logger.info("=" * 80)
    logger.info("損失組成:")
    logger.info(f"  1. CrossEntropy Loss (Token 準確度)")
    logger.info(f"     - 權重: {args.ce_weight} (固定)")
    logger.info(f"  2. Content Consistency Loss (相同內容應相似)")
    logger.info(f"     - 最大權重: {args.content_weight}")
    logger.info(f"     - 動態調整: Epoch 0-{args.warmup_epochs} warmup, 之後衰減")
    logger.info(f"  3. Embedding L2 Loss (Embedding 空間接近)")
    logger.info(f"     - 權重: {args.embed_weight} (固定)")
    logger.info("")
    logger.info("動態權重調度:")
    logger.info(f"  - Epoch 0-{args.warmup_epochs}: 權重從 {args.content_weight} → {args.content_weight*0.5} (線性)")
    logger.info(f"  - Epoch {args.warmup_epochs}+: 權重從 {args.content_weight*0.5} → ~0.01 (指數衰減)")
    logger.info("=" * 80)
    
    criterion = DiscreteHybridLoss(
        codebook=codebook,
        wavtokenizer=None,  # 不使用 spectral loss
        device=device,
        ce_weight=args.ce_weight,
        content_weight=args.content_weight,
        embed_weight=args.embed_weight,
        spectral_weight=0.0,  # 關閉
        warmup_epochs=args.warmup_epochs
    )
    
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
            model, train_loader, optimizer, criterion, device, epoch, args.num_epochs
        )
        
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"  Train - Total Loss: {train_metrics['total_loss']:.4f}, "
                   f"CE: {train_metrics['ce_loss']:.4f}, "
                   f"Content: {train_metrics['content_loss']:.4f}, "
                   f"Embed: {train_metrics['embed_loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.2f}%")
        
        # 驗證
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, args.num_epochs
        )
        
        logger.info(f"  Val   - Total Loss: {val_metrics['total_loss']:.4f}, "
                   f"CE: {val_metrics['ce_loss']:.4f}, "
                   f"Content: {val_metrics['content_loss']:.4f}, "
                   f"Embed: {val_metrics['embed_loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.2f}%")
        
        # 記錄訓練歷史
        train_loss_history.append(train_metrics)
        val_loss_history.append(val_metrics)
        train_acc_history.append(train_metrics['accuracy'])
        val_acc_history.append(val_metrics['accuracy'])
        
        # 更新學習率 - 根據驗證損失調整
        scheduler.step(val_metrics['total_loss'])
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        
        # 保存 checkpoint (每 10 epochs)
        if epoch % 10 == 0:
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
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
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
            logger.info(f"  保存音頻樣本...")
            try:
                # 從驗證集取一個 batch
                val_batch = next(iter(val_loader))
                noisy_tokens, clean_tokens, _ = val_batch
                noisy_tokens = noisy_tokens.to(device)
                clean_tokens = clean_tokens.to(device)
                
                # 預測
                model.eval()
                with torch.no_grad():
                    pred_logits = model(noisy_tokens, return_logits=True)
                    pred_tokens = pred_logits.argmax(dim=-1)
                
                # 保存音頻和頻譜圖
                save_audio_samples(
                    wavtokenizer=wavtokenizer,
                    noisy_tokens=noisy_tokens,
                    pred_tokens=pred_tokens,
                    clean_tokens=clean_tokens,
                    epoch=epoch,
                    output_dir=output_dir,
                    device=device,
                    num_samples=3
                )
                logger.info(f"  ✓ 已保存音頻樣本到: audio_samples/epoch_{epoch}/")
                model.train()
            except Exception as e:
                logger.error(f"  保存音頻樣本時出錯: {e}")
    
    logger.info("訓練完成！")
    logger.info(f"最佳驗證損失: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

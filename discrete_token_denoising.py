#!/usr/bin/env python3
"""
基於離散特徵的 Token-to-Token 降噪訓練腳本
使用 Transformer Encoder-Decoder 架構進行序列到序列的降噪學習

架構說明：
- Encoder: 理解帶噪 token 序列的全局模式
- Decoder: 在 Encoder 指導下生成乾淨的 token 序列
- 使用交叉熵損失進行訓練
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import logging
import math
from typing import Tuple, Optional, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder.pretrained import WavTokenizer
from ttdata import AudioDataset
from token_loss_system import compute_combined_token_loss

def set_seed(seed=42):
    """設定隨機種子以確保實驗可重現"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TokenSequenceDataset(Dataset):
    """Token 序列資料集類別"""
    
    def __init__(self, noisy_tokens, clean_tokens, max_length=512):
        """
        Args:
            noisy_tokens: List[torch.Tensor] - 帶噪 token 序列列表
            clean_tokens: List[torch.Tensor] - 乾淨 token 序列列表
            max_length: int - 最大序列長度
        """
        assert len(noisy_tokens) == len(clean_tokens), "Noisy and clean token lists must have same length"
        self.noisy_tokens = noisy_tokens
        self.clean_tokens = clean_tokens
        self.max_length = max_length
        
        # 特殊 tokens
        self.pad_token = 0
        self.sos_token = 4096  # Start of sequence
        self.eos_token = 4097  # End of sequence
        self.vocab_size = 4098  # codebook_size + special_tokens
    
    def __len__(self):
        return len(self.noisy_tokens)
    
    def __getitem__(self, idx):
        noisy_seq = self.noisy_tokens[idx].flatten()
        clean_seq = self.clean_tokens[idx].flatten()
        
        # 截斷過長序列
        if len(noisy_seq) > self.max_length - 1:  # 留空間給 EOS
            noisy_seq = noisy_seq[:self.max_length - 1]
        if len(clean_seq) > self.max_length - 2:  # 留空間給 SOS, EOS
            clean_seq = clean_seq[:self.max_length - 2]
        
        # 確保數據類型正確
        noisy_seq = noisy_seq.long()
        clean_seq = clean_seq.long()
        
        # 準備輸入序列 (noisy + EOS)
        input_seq = torch.cat([noisy_seq, torch.tensor([self.eos_token], dtype=torch.long)])
        
        # 準備目標序列 (SOS + clean + EOS)
        target_seq = torch.cat([torch.tensor([self.sos_token], dtype=torch.long), clean_seq, torch.tensor([self.eos_token], dtype=torch.long)])
        
        # 準備解碼器輸入 (SOS + clean，用於 teacher forcing)
        decoder_input = torch.cat([torch.tensor([self.sos_token], dtype=torch.long), clean_seq])
        
        # Padding
        input_length = len(input_seq)
        target_length = len(target_seq)
        decoder_input_length = len(decoder_input)
        
        if input_length < self.max_length:
            input_seq = torch.cat([input_seq, torch.zeros(self.max_length - input_length, dtype=torch.long)])
        
        if target_length < self.max_length:
            target_seq = torch.cat([target_seq, torch.zeros(self.max_length - target_length, dtype=torch.long)])
        
        if decoder_input_length < self.max_length:
            decoder_input = torch.cat([decoder_input, torch.zeros(self.max_length - decoder_input_length, dtype=torch.long)])
        
        return {
            'input_seq': input_seq,
            'decoder_input': decoder_input,
            'target_seq': target_seq,
            'input_length': input_length,
            'target_length': target_length
        }

class PositionalEncoding(nn.Module):
    """位置編碼"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TokenToTokenTransformer(nn.Module):
    """Token-to-Token Transformer 降噪模型"""
    
    def __init__(self, vocab_size=4098, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, max_length=512, dropout=0.1):
        super(TokenToTokenTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Token embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型參數"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """生成解碼器的因果遮罩"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def create_padding_mask(self, seq, pad_token=0):
        """創建 padding 遮罩"""
        return (seq == pad_token)
    
    def forward(self, src, tgt, src_length=None, tgt_length=None):
        """
        Args:
            src: [batch_size, src_seq_len] - 輸入序列 (noisy tokens)
            tgt: [batch_size, tgt_seq_len] - 目標序列 (用於 teacher forcing)
            src_length: [batch_size] - 輸入序列實際長度
            tgt_length: [batch_size] - 目標序列實際長度
        """
        batch_size, src_seq_len = src.size()
        _, tgt_seq_len = tgt.size()
        
        # Embeddings
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)  # [batch_size, src_seq_len, d_model]
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, tgt_seq_len, d_model]
        
        # Positional encoding
        src_emb = src_emb.transpose(0, 1)  # [src_seq_len, batch_size, d_model]
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_seq_len, batch_size, d_model]
        
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        src_emb = src_emb.transpose(0, 1)  # [batch_size, src_seq_len, d_model]
        tgt_emb = tgt_emb.transpose(0, 1)  # [batch_size, tgt_seq_len, d_model]
        
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )  # [batch_size, tgt_seq_len, d_model]
        
        # Output projection
        logits = self.output_projection(output)  # [batch_size, tgt_seq_len, vocab_size]
        
        return logits

def extract_token_sequences(audio_dataset, wavtokenizer, device, max_samples=None):
    """
    從音頻資料集中提取 token 序列
    
    Args:
        audio_dataset: AudioDataset 實例
        wavtokenizer: WavTokenizer 模型
        device: 計算設備
        max_samples: 最大處理樣本數
    
    Returns:
        noisy_tokens: List[torch.Tensor] - 帶噪 token 序列
        clean_tokens: List[torch.Tensor] - 乾淨 token 序列
        speakers: List[str] - 對應的語者信息
        content_ids: List[str] - 對應的內容ID
    """
    logging.info("開始提取 token 序列...")
    
    wavtokenizer.eval()
    noisy_tokens = []
    clean_tokens = []
    speakers = []
    content_ids = []
    
    dataloader = DataLoader(audio_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="提取 tokens")):
            if max_samples and i >= max_samples:
                break
            
            # AudioDataset 返回 (input_wav, target_wav, content_id)
            if len(batch) == 3:
                noisy_audio, clean_audio, content_id = batch
                content_id = content_id[0] if isinstance(content_id, list) else content_id
            else:
                # 如果是字典格式（備用）
                noisy_audio = batch['noisy_audio'] if 'noisy_audio' in batch else batch['input_audio']
                clean_audio = batch['clean_audio'] if 'clean_audio' in batch else batch['target_audio']
                content_id = "unknown"
            
            noisy_audio = noisy_audio.to(device)
            clean_audio = clean_audio.to(device)
            
            # 從 content_id 或配對文件信息中提取語者信息
            try:
                # 獲取當前樣本的配對信息
                if hasattr(audio_dataset, 'paired_files') and i < len(audio_dataset.paired_files):
                    speaker = audio_dataset.paired_files[i]['speaker']
                else:
                    # 從 content_id 中提取語者（例如：從 "137" 推斷或使用文件名模式）
                    speaker = "unknown"
            except:
                speaker = "unknown"
            
            # 提取 noisy tokens
            try:
                bandwidth_id = torch.tensor([0], device=device)  # 最高質量編碼
                noisy_tokens_batch, _ = wavtokenizer.encode_infer(noisy_audio, bandwidth_id=bandwidth_id)
                if isinstance(noisy_tokens_batch, tuple):
                    noisy_tokens_batch = noisy_tokens_batch[0]  # 取第一個量化層
                noisy_tokens.append(noisy_tokens_batch.squeeze(0).cpu())
            except Exception as e:
                logging.warning(f"提取 noisy tokens 失敗 (batch {i}): {e}")
                continue
            
            # 提取 clean tokens
            try:
                clean_tokens_batch, _ = wavtokenizer.encode_infer(clean_audio, bandwidth_id=bandwidth_id)
                if isinstance(clean_tokens_batch, tuple):
                    clean_tokens_batch = clean_tokens_batch[0]  # 取第一個量化層
                clean_tokens.append(clean_tokens_batch.squeeze(0).cpu())
                
                # 保存語者和內容ID信息
                speakers.append(speaker)
                content_ids.append(content_id)
                
            except Exception as e:
                logging.warning(f"提取 clean tokens 失敗 (batch {i}): {e}")
                noisy_tokens.pop()  # 移除對應的 noisy token
                continue
    
    logging.info(f"成功提取 {len(noisy_tokens)} 對 token 序列")
    logging.info(f"語者分佈: {set(speakers)}")
    return noisy_tokens, clean_tokens, speakers, content_ids

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """使用原始 CrossEntropy 訓練一個 epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_seq = batch['input_seq'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        target_seq = batch['target_seq'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_seq, decoder_input)  # [batch_size, seq_len, vocab_size]
        
        # 計算損失 (忽略 padding tokens)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target_seq.view(-1)
        
        loss = criterion(logits_flat, target_flat)
        
        # Backward pass
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 統計
        total_loss += loss.item()
        total_tokens += (target_seq != 0).sum().item()  # 計算非 padding tokens
        
        # 更新進度條
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)

def train_epoch_with_token_loss(model, dataloader, optimizer, device, epoch, 
                               loss_weights={'l2': 0.3, 'consistency': 0.4, 'manifold': 0.1, 
                                           'normalization': 0.1, 'coherence': 0.1}):
    """使用 Token Loss 系統訓練一個 epoch（按照 ttt2.py 的 loss 邏輯）"""
    model.train()
    total_losses = {'total': 0.0}
    loss_counts = 0
    
    # 獲取嵌入層
    embedding_layer = None
    if hasattr(model, 'src_embedding'):
        embedding_layer = model.src_embedding  # TokenToTokenTransformer 的源嵌入層
    elif hasattr(model, 'tgt_embedding'):
        embedding_layer = model.tgt_embedding  # TokenToTokenTransformer 的目標嵌入層
    elif hasattr(model, 'embedding'):
        embedding_layer = model.embedding
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'embeddings'):
        embedding_layer = model.transformer.embeddings.word_embeddings
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'embed_tokens'):
        embedding_layer = model.encoder.embed_tokens
    
    if embedding_layer is None:
        logging.warning("無法找到嵌入層，將使用簡化的 token loss")
    else:
        logging.info(f"找到嵌入層：{type(embedding_layer).__name__}, 嵌入維度：{embedding_layer.embedding_dim}")
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (Token Loss)")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_seq = batch['input_seq'].to(device)
        decoder_input = batch['decoder_input'].to(device) 
        target_seq = batch['target_seq'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_seq, decoder_input)  # [batch_size, seq_len, vocab_size]
        predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        
        # 使用 Token Loss 系統計算損失
        try:
            total_loss, loss_dict = compute_combined_token_loss(
                predicted_logits=logits,
                predicted_tokens=predicted_tokens,
                target_tokens=target_seq,
                input_tokens=input_seq,
                embedding_layer=embedding_layer,
                weights=loss_weights
            )
        except Exception as e:
            logging.warning(f"Token loss 計算失敗，回退到交叉熵: {e}")
            # 回退到簡單交叉熵
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_seq.view(-1)
            total_loss = F.cross_entropy(logits_flat, target_flat, ignore_index=0)
            loss_dict = {'total_loss': total_loss.item(), 'consistency_loss': total_loss.item()}
        
        # Backward pass
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累積統計
        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value
        
        loss_counts += 1
        
        # 更新進度條
        avg_total_loss = total_losses.get('total_loss', total_losses['total']) / loss_counts
        progress_info = {'Total': f'{avg_total_loss:.4f}'}
        
        # 添加主要 loss 組件到進度條
        if 'consistency_loss' in total_losses:
            progress_info['Consistency'] = f'{total_losses["consistency_loss"]/loss_counts:.4f}'
        if 'l2_loss' in total_losses:
            progress_info['L2'] = f'{total_losses["l2_loss"]/loss_counts:.4f}'
        
        progress_bar.set_postfix(progress_info)
    
    # 計算平均損失
    avg_losses = {key: value / loss_counts for key, value in total_losses.items()}
    
    return avg_losses

def validate_epoch(model, dataloader, criterion, device):
    """驗證一個 epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_seq = batch['input_seq'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            target_seq = batch['target_seq'].to(device)
            
            # Forward pass
            logits = model(input_seq, decoder_input)
            
            # 計算損失
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_seq.view(-1)
            
            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item()
            
            # 計算準確率 (忽略 padding tokens)
            predictions = torch.argmax(logits_flat, dim=-1)
            mask = (target_flat != 0)  # 非 padding tokens
            total_correct += ((predictions == target_flat) & mask).sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, loss, save_path, model_config=None):
    """保存模型檢查點"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model_config,
    }
    torch.save(checkpoint, save_path)
    logging.info(f"檢查點已保存到: {save_path}")

def plot_training_history(train_losses, val_losses, val_accuracies, save_path):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logging.info(f"訓練歷史圖已保存到: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Token-to-Token 降噪 Transformer 訓練')
    
    # 基本參數
    parser.add_argument('--config', type=str, default='config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                        help='WavTokenizer 配置文件路徑')
    parser.add_argument('--model_path', type=str, default='models/wavtokenizer_large_speech_320_24k.ckpt',
                        help='WavTokenizer 預訓練模型路徑')
    parser.add_argument('--output_dir', type=str, default='results/discrete_token_denoising',
                        help='輸出目錄')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='最大處理樣本數 (用於快速測試)')
    
    # 模型參數
    parser.add_argument('--d_model', type=int, default=512, help='模型維度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力頭數')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='編碼器層數')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='解碼器層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='前饋網絡維度')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列長度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 率')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=300, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='權重衰減')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='預熱步數')
    parser.add_argument('--save_every', type=int, default=10, help='每幾個 epoch 保存一次')
    
    # 損失函數選擇
    parser.add_argument('--use_token_loss', action='store_true', 
                        help='使用 Token Loss 系統（按照 ttt2.py 邏輯）而非單純交叉熵')
    parser.add_argument('--l2_weight', type=float, default=0.3, help='L2 距離損失權重')
    parser.add_argument('--consistency_weight', type=float, default=0.4, help='內容一致性損失權重')
    parser.add_argument('--manifold_weight', type=float, default=0.1, help='Manifold 正則化權重')
    parser.add_argument('--normalization_weight', type=float, default=0.1, help='正則化損失權重')
    parser.add_argument('--coherence_weight', type=float, default=0.1, help='連貫性損失權重')
    
    # 資料參數
    parser.add_argument('--train_split', type=float, default=0.8, help='訓練集比例')
    parser.add_argument('--max_sentences_per_speaker', type=int, default=100,
                        help='每位語者最大句子數')
    
    # 驗證集參數（與 ttt2.py 保持一致）
    parser.add_argument('--validation_strategy', type=str, default='speaker_only',
                        choices=['random', 'speaker_only'], help='驗證集分割策略')
    parser.add_argument('--custom_val_split', action='store_true', default=True,
                        help='啟用自定義驗證集分割')
    parser.add_argument('--val_speakers', nargs='+', default=['girl9', 'boy7'],
                        help='指定驗證集語者（與 ttt2.py 相同）')
    
    args = parser.parse_args()
    
    # 設定隨機種子
    set_seed(42)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用設備: {device}")
    
    # 載入 WavTokenizer
    logging.info("載入 WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(args.config, args.model_path)
    wavtokenizer = wavtokenizer.to(device)
    
    # 準備音頻資料集
    logging.info("準備音頻資料集...")
    logging.info("="*60)
    logging.info("數據配置 (與 ttt2.py 保持一致):")
    logging.info("  • 10位訓練語者 + 2位驗證語者")
    logging.info("  • 每位語者最多100句話 (句子編號1-100)")
    logging.info("  • 僅使用box材質數據")
    logging.info("  • 驗證語者: girl9, boy7")
    logging.info("="*60)
    
    # 使用與 ttt2.py 相同的數據目錄設定
    input_dirs = [
        os.path.join(os.getcwd(), "data", "raw", "box")  # 僅使用 box 材質
    ]
    target_dir = os.path.join(os.getcwd(), "data", "clean", "box2")
    
    # 如果數據目錄不存在，使用基本初始化
    if not any(os.path.exists(d) for d in input_dirs) or not os.path.exists(target_dir):
        logging.warning("未找到標準數據目錄，嘗試使用基本 AudioDataset 初始化")
        try:
            audio_dataset = AudioDataset()
        except TypeError:
            # 如果還是失敗，創建一個虛擬數據集用於測試
            logging.warning("AudioDataset 初始化失敗，創建虛擬數據集進行測試")
            noisy_tokens = [torch.randint(0, 4096, (100,)) for _ in range(10)]
            clean_tokens = [torch.randint(0, 4096, (95,)) for _ in range(10)]
            
            # 為虛擬數據集創建對應的語者和內容ID信息
            speakers = ['virtual_speaker_' + str(i % 2) for i in range(10)]  # 創建2個虛擬語者
            content_ids = ['virtual_content_' + str(i) for i in range(10)]
            
            # 創建 token 序列資料集
            logging.info("使用虛擬 token 數據創建資料集...")
            token_dataset = TokenSequenceDataset(noisy_tokens, clean_tokens, max_length=args.max_length)
        else:
            # 提取 token 序列
            noisy_tokens, clean_tokens, speakers, content_ids = extract_token_sequences(
                audio_dataset, wavtokenizer, device, max_samples=args.max_samples
            )
            
            if len(noisy_tokens) == 0:
                logging.error("未能提取任何 token 序列，請檢查資料集和模型")
                return
            
            # 創建 token 序列資料集
            logging.info("創建 token 序列資料集...")
            token_dataset = TokenSequenceDataset(noisy_tokens, clean_tokens, max_length=args.max_length)
    else:
        logging.info(f"創建AudioDataset，限制每位語者最多 {args.max_sentences_per_speaker} 句話...")
        audio_dataset = AudioDataset(input_dirs, target_dir, max_sentences_per_speaker=args.max_sentences_per_speaker)
        
        # 驗證數據集大小
        logging.info("="*50)
        logging.info("數據集統計:")
        logging.info(f"  實際配對文件數: {len(audio_dataset)} 個")
        logging.info(f"  理論最大文件數: 12位語者 × {args.max_sentences_per_speaker}句/語者 = {12 * args.max_sentences_per_speaker} 個")
        logging.info(f"  數據完整度: {len(audio_dataset)/(12 * args.max_sentences_per_speaker)*100:.1f}%")
        logging.info("="*50)
        
        # 提取 token 序列 - 使用所有可用數據（不限制樣本數）
        noisy_tokens, clean_tokens, speakers, content_ids = extract_token_sequences(
            audio_dataset, wavtokenizer, device, max_samples=None
        )
        
        if len(noisy_tokens) == 0:
            logging.error("未能提取任何 token 序列，請檢查資料集和模型")
            return

        # 詳細統計信息
        logging.info("="*50)
        logging.info("Token提取結果:")
        logging.info(f"  成功提取: {len(noisy_tokens)} 對 token 序列")
        logging.info(f"  語者總數: {len(set(speakers))} 位")
        logging.info(f"  語者列表: {sorted(set(speakers))}")
        
        # 統計每個語者的句子數和編號範圍
        from collections import Counter, defaultdict
        speaker_counts = Counter(speakers)
        speaker_content_ids = defaultdict(list)
        
        # 收集每個語者的內容ID
        for speaker, content_id in zip(speakers, content_ids):
            try:
                content_num = int(content_id)
                speaker_content_ids[speaker].append(content_num)
            except (ValueError, TypeError):
                pass  # 忽略無法轉換為整數的ID
        
        logging.info("  每位語者的句子數和編號範圍:")
        sentence_range_violations = []
        for speaker, count in sorted(speaker_counts.items()):
            if speaker in speaker_content_ids:
                content_nums = sorted(speaker_content_ids[speaker])
                if content_nums:
                    min_id, max_id = min(content_nums), max(content_nums)
                    # 檢查是否超出1-100範圍
                    if min_id < 1 or max_id > 100:
                        sentence_range_violations.append((speaker, min_id, max_id))
                        logging.warning(f"    {speaker}: {count} 句 (編號範圍: {min_id}-{max_id}) ⚠ 超出1-100範圍")
                    else:
                        logging.info(f"    {speaker}: {count} 句 (編號範圍: {min_id}-{max_id}) ✓")
                else:
                    logging.info(f"    {speaker}: {count} 句 (無有效編號)")
            else:
                logging.info(f"    {speaker}: {count} 句")
        
        # 報告編號範圍檢查結果
        if sentence_range_violations:
            logging.warning(f"發現 {len(sentence_range_violations)} 位語者使用超出1-100範圍的句子編號")
        else:
            logging.info("✓ 所有語者都使用編號1-100範圍內的句子")
        logging.info("="*50)
        
        # 創建 token 序列資料集
        logging.info("創建 token 序列資料集...")
        token_dataset = TokenSequenceDataset(noisy_tokens, clean_tokens, max_length=args.max_length)
    
    # 按語者分割訓練集和驗證集（與 ttt2.py 保持一致）
    if args.custom_val_split and args.validation_strategy == 'speaker_only':
        logging.info(f"使用自定義驗證集分割，驗證語者: {args.val_speakers}")
        
        # 分離訓練和驗證樣本的索引
        train_indices = []
        val_indices = []
        
        # 檢查可用的語者
        available_speakers = set(speakers)
        requested_val_speakers = set(args.val_speakers)
        actual_val_speakers = available_speakers.intersection(requested_val_speakers)
        
        if not actual_val_speakers:
            logging.warning(f"未找到請求的驗證語者 {requested_val_speakers}，可用語者: {available_speakers}")
            logging.warning("將使用可用語者中的前幾個作為驗證語者")
            actual_val_speakers = set(list(available_speakers)[:min(2, len(available_speakers))])
        
        logging.info(f"實際使用的驗證語者: {actual_val_speakers}")
        
        for i, speaker in enumerate(speakers):
            if speaker in actual_val_speakers:
                val_indices.append(i)
            else:
                train_indices.append(i)
        
        logging.info("="*50)
        logging.info("數據集分割結果 (按語者分割，與ttt2.py一致):")
        training_speakers = set(speakers[i] for i in train_indices)
        validation_speakers = set(speakers[i] for i in val_indices)
        
        logging.info(f"  訓練集:")
        logging.info(f"    樣本數: {len(train_indices)} 個")
        logging.info(f"    語者數: {len(training_speakers)} 位")
        logging.info(f"    語者列表: {sorted(training_speakers)}")
        
        logging.info(f"  驗證集:")
        logging.info(f"    樣本數: {len(val_indices)} 個")
        logging.info(f"    語者數: {len(validation_speakers)} 位")
        logging.info(f"    語者列表: {sorted(validation_speakers)}")
        
        # 驗證分割的正確性
        if validation_speakers == actual_val_speakers:
            logging.info("  ✓ 驗證語者配置正確")
        else:
            logging.warning(f"  ⚠ 驗證語者配置異常: 預期 {actual_val_speakers}, 實際 {validation_speakers}")
        logging.info("="*50)
        
        # 創建子集
        train_dataset = torch.utils.data.Subset(token_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(token_dataset, val_indices)
        
    else:
        # 使用隨機分割（備用方案）
        logging.info(f"使用隨機分割，比例: {args.train_split}")
        train_size = int(args.train_split * len(token_dataset))
        val_size = len(token_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(token_dataset, [train_size, val_size])
    
    # 創建資料載入器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logging.info(f"訓練集大小: {len(train_dataset)}, 驗證集大小: {len(val_dataset)}")
    
    # 創建模型
    logging.info("創建 Transformer 模型...")
    model = TokenToTokenTransformer(
        vocab_size=token_dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_length=args.max_length,
        dropout=args.dropout
    ).to(device)
    
    # 計算模型參數數量
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"模型總參數數量: {total_params:,}")
    
    # 創建優化器和損失函數
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding tokens
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs,
        pct_start=0.1
    )
    
        # 準備損失權重字典
    loss_weights = {
        'l2': args.l2_weight,
        'consistency': args.consistency_weight,
        'manifold': args.manifold_weight,
        'normalization': args.normalization_weight,
        'coherence': args.coherence_weight
    }
    
    # 訓練循環
    logging.info("開始訓練...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # 選擇訓練函數
        if args.use_token_loss:
            # 使用 Token Loss 系統（按照 ttt2.py 邏輯）
            train_loss_dict = train_epoch_with_token_loss(
                model, train_loader, optimizer, device, epoch, loss_weights
            )
            train_loss = train_loss_dict.get('total_loss', 0.0)
            
            # 記錄詳細的損失信息
            loss_info = " | ".join([f"{k}: {v:.4f}" for k, v in train_loss_dict.items() 
                                   if k != 'total_loss'])
            logging.info(f"Epoch {epoch} Train Loss Components: {loss_info}")
        else:
            # 使用原始交叉熵
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        train_losses.append(train_loss)
        
        # 驗證（始終使用原始方法以便比較）
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 學習率調度
        scheduler.step()
        
        # 記錄
        logging.info(f"Epoch {epoch}/{args.num_epochs}")
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        logging.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        logging.info(f"Loss System: {'Token Loss (ttt2.py style)' if args.use_token_loss else 'Cross Entropy'}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_config = {
                'vocab_size': model.vocab_size,
                'd_model': model.d_model,
                'nhead': model.transformer.nhead,
                'num_encoder_layers': len(model.transformer.encoder.layers),
                'num_decoder_layers': len(model.transformer.decoder.layers),
                'dim_feedforward': args.dim_feedforward,  # 直接使用訓練參數
                'max_length': model.max_length,
                'dropout': args.dropout,  # 直接使用訓練參數
            }
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, 'best_model.pth'),
                model_config
            )
            logging.info(f"新的最佳模型已保存 (Val Loss: {val_loss:.4f})")
        
        # 定期保存
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'),
                model_config
            )
        
        # 繪製訓練歷史
        if epoch % 5 == 0:
            plot_training_history(
                train_losses, val_losses, val_accuracies,
                os.path.join(args.output_dir, 'training_history.png')
            )
    
    logging.info("訓練完成！")
    
    # 最終保存
    final_model_config = {
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'nhead': model.transformer.nhead,
        'num_encoder_layers': len(model.transformer.encoder.layers),
        'num_decoder_layers': len(model.transformer.decoder.layers),
        'dim_feedforward': args.dim_feedforward,  # 直接使用訓練參數
        'max_length': model.max_length,
        'dropout': args.dropout,  # 直接使用訓練參數
    }
    save_checkpoint(
        model, optimizer, args.num_epochs, val_losses[-1],
        os.path.join(args.output_dir, 'final_model.pth'),
        final_model_config
    )
    
    plot_training_history(
        train_losses, val_losses, val_accuracies,
        os.path.join(args.output_dir, 'final_training_history.png')
    )

if __name__ == "__main__":
    main()

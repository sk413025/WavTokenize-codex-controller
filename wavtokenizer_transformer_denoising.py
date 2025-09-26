#!/usr/bin/env python3
"""
基於 WavTokenizer 的端到端音頻降噪系統
架構：Audio → WavTokenizer Encoder (凍結) → Tokens → Transformer (可訓練) → Denoised Tokens → WavTokenizer Decoder (凍結) → Audio

核心理念：
- 使用預訓練的 WavTokenizer Encoder 將音頻轉換為 discrete tokens
- 使用 Transformer 在 token 空間進行降噪學習
- 使用預訓練的 WavTokenizer Decoder 將 denoised tokens 重建為音頻
- 只有中間的 Transformer 部分需要訓練，WavTokenizer 的 Encoder/Decoder 保持凍結
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
import torchaudio
import random

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder.pretrained import WavTokenizer
from ttdata import AudioDataset
from ttt2 import collate_fn
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

def plot_spectrograms(audio, save_path, device, title="Spectrogram"):
    """繪製並保存頻譜圖（與 ttt2.py 一致）"""
    try:
        # 確保音頻在CPU上並轉為numpy
        if isinstance(audio, torch.Tensor):
            if audio.is_cuda:
                audio = audio.cpu()
            audio = audio.squeeze().numpy()
        
        # 創建梅爾頻譜圖變換
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=512,
            n_mels=80,
            f_min=20,
            f_max=8000
        )
        
        # 轉回tensor並計算頻譜圖
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        mel_spec = transform(audio_tensor)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # 繪製
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spec_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='dB')
        plt.title(title, fontsize=14)
        plt.xlabel('Time frames')
        plt.ylabel('Mel frequency bins')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in plot_spectrograms: {str(e)}")

def save_audio_sample(audio, save_path, sample_rate=24000):
    """保存音頻樣本"""
    try:
        # 確保音頻格式正確
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 3:
                audio = audio.squeeze(0)
        
        # 正規化音頻
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
        
        torchaudio.save(save_path, audio.cpu(), sample_rate)
    except Exception as e:
        print(f"Error saving audio: {str(e)}")

def save_sample_with_spectrograms(input_audio, output_audio, target_audio, epoch, batch_idx, save_dir, prefix="sample"):
    """保存音頻樣本及其頻譜圖（與 ttt2.py 一致）"""
    try:
        # 創建樣本目錄
        sample_dir = os.path.join(save_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存音頻文件
        audio_files = {
            'input': input_audio,
            'output': output_audio, 
            'target': target_audio
        }
        
        for audio_type, audio_data in audio_files.items():
            if audio_data is not None:
                # 保存音頻
                audio_path = os.path.join(sample_dir, f'{prefix}_epoch{epoch}_batch{batch_idx}_{audio_type}.wav')
                save_audio_sample(audio_data, audio_path)
                
                # 保存頻譜圖
                spec_path = os.path.join(sample_dir, f'{prefix}_epoch{epoch}_batch{batch_idx}_{audio_type}_spec.png')
                plot_spectrograms(audio_data, spec_path, 'cpu', f"Epoch {epoch} {audio_type.capitalize()} Spectrogram")
        
        print(f"Saved sample for epoch {epoch}, batch {batch_idx}")
        
    except Exception as e:
        print(f"Error saving sample: {str(e)}")

class WavTokenizerTransformerDenoiser(nn.Module):
    """基於 WavTokenizer 的端到端音頻降噪系統"""
    
    def __init__(self, config_path, model_path, 
                 d_model=128, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=256, max_length=256, dropout=0.1):
        super(WavTokenizerTransformerDenoiser, self).__init__()
        
        # 載入預訓練的 WavTokenizer
        self.wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
        
        # 凍結 WavTokenizer 的所有參數
        for param in self.wavtokenizer.parameters():
            param.requires_grad = False
        
        # Token 相關設定
        self.codebook_size = 4096  # WavTokenizer 的 codebook 大小
        self.pad_token = self.codebook_size  # padding token
        self.sos_token = self.codebook_size + 1  # start of sequence
        self.eos_token = self.codebook_size + 2  # end of sequence
        self.vocab_size = self.codebook_size + 3  # total vocabulary size
        
        # Transformer 降噪器參數
        self.d_model = d_model
        self.max_length = max_length
        
        # Token embeddings (僅用於 Transformer 部分)
        self.src_embedding = nn.Embedding(self.vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(self.vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(d_model, max_length)
        
        # Transformer 降噪器
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection (將 Transformer 輸出投影回 token 空間)
        self.output_projection = nn.Linear(d_model, self.codebook_size)
        
        # 初始化參數
        self._init_parameters()
    
    def _create_positional_encoding(self, d_model, max_len=5000):
        """創建位置編碼"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        return nn.Parameter(pe, requires_grad=False)
    
    def _init_parameters(self):
        """初始化模型參數"""
        for p in self.parameters():
            if p.requires_grad and p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode_audio_to_tokens(self, audio):
        """使用 WavTokenizer Encoder 將音頻轉換為 tokens"""
        with torch.no_grad():
            bandwidth_id = torch.tensor([0], device=audio.device)  # 使用最高質量
            _, discrete_codes = self.wavtokenizer.encode_infer(audio, bandwidth_id=bandwidth_id)
            
            # discrete_codes 形狀通常是 [n_q, batch_size, seq_len] 或 [batch_size, n_q, seq_len]
            if discrete_codes.dim() == 3:
                # 使用第一個量化層的 tokens
                tokens = discrete_codes[0] if discrete_codes.size(0) == 1 else discrete_codes[:, 0, :]
            else:
                tokens = discrete_codes
            
            tokens = tokens.long()
            # 確保tokens在詞彙範圍內
            tokens = torch.clamp(tokens, 0, self.codebook_size - 1)
            return tokens
    
    def decode_tokens_to_audio(self, tokens):
        """使用 WavTokenizer Decoder 將 tokens 轉換為音頻"""
        with torch.no_grad():
            # 首先將 tokens 轉換為 features
            # 需要將 tokens 轉換為正確的形狀：[n_q, batch_size, seq_len]
            if tokens.dim() == 2:  # [batch_size, seq_len]
                tokens = tokens.unsqueeze(0)  # [1, batch_size, seq_len]
            
            # 使用 codes_to_features 將 discrete codes 轉換為連續特徵
            features = self.wavtokenizer.codes_to_features(tokens)
            
            # 使用 decoder 將特徵重建為音頻
            bandwidth_id = torch.tensor([0], device=tokens.device)
            audio = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
            
            # WavTokenizer.decode 返回 [batch, time]，需要添加 channel 維度變為 [batch, 1, time]
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # [batch, time] -> [batch, 1, time]
            
            # 如果是4D張量，進行維度調整
            elif audio.dim() == 4:
                if audio.size(1) == 1:
                    audio = audio.squeeze(1)  # [batch, 1, 1, time] -> [batch, 1, time]
                elif audio.size(2) == 1:
                    audio = audio.squeeze(2)  # [batch, channels, 1, time] -> [batch, channels, time]
            
            return audio
    
    def generate_square_subsequent_mask(self, sz):
        """生成解碼器的因果遮罩"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def create_padding_mask(self, seq, pad_token=None):
        """創建 padding 遮罩"""
        if pad_token is None:
            pad_token = self.pad_token
        return (seq == pad_token)
    
    def forward_transformer(self, src_tokens, tgt_tokens=None):
        """Transformer 前向傳播（僅處理 token 序列）"""
        batch_size, src_seq_len = src_tokens.size()
        max_pos_len = self.pos_encoding.shape[1]
        
        # 處理 src_tokens 的長度
        if src_seq_len < max_pos_len:
            # 填充到最大長度
            pad_size = max_pos_len - src_seq_len
            pad = torch.full((batch_size, pad_size), self.pad_token, device=src_tokens.device, dtype=src_tokens.dtype)
            src_tokens_padded = torch.cat([src_tokens, pad], dim=1)
        elif src_seq_len > max_pos_len:
            # 裁切到最大長度
            src_tokens_padded = src_tokens[:, :max_pos_len]
        else:
            src_tokens_padded = src_tokens
            
        # 使用處理後的 tokens 計算 embedding
        src_emb = self.src_embedding(src_tokens_padded) * math.sqrt(self.d_model)
        src_seq_len = src_tokens_padded.shape[1]
        src_emb = src_emb + self.pos_encoding[:, :src_seq_len, :]
        
        # 如果是訓練模式且提供了目標序列
        if tgt_tokens is not None and self.training:
            _, tgt_seq_len = tgt_tokens.size()
            max_pos_len = self.pos_encoding.shape[1]
            tgt_emb = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
            if tgt_seq_len < max_pos_len:
                pad_size = max_pos_len - tgt_seq_len
                pad = torch.zeros(tgt_emb.shape[0], pad_size, tgt_emb.shape[2], device=tgt_emb.device, dtype=tgt_emb.dtype)
                tgt_emb = torch.cat([tgt_emb, pad], dim=1)
                tgt_seq_len = max_pos_len
            elif tgt_seq_len > max_pos_len:
                tgt_emb = tgt_emb[:, :max_pos_len, :]
                tgt_seq_len = max_pos_len
            tgt_emb = tgt_emb + self.pos_encoding[:, :tgt_seq_len, :]
            # 創建遮罩
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src_tokens.device)
            src_padding_mask = self.create_padding_mask(src_tokens_padded)
            tgt_padding_mask = self.create_padding_mask(tgt_tokens)
            # 填充或裁切 tgt_padding_mask 到 max_pos_len
            if tgt_padding_mask.shape[1] < max_pos_len:
                pad_size = max_pos_len - tgt_padding_mask.shape[1]
                pad = torch.zeros(tgt_padding_mask.shape[0], pad_size, device=tgt_padding_mask.device, dtype=tgt_padding_mask.dtype)
                tgt_padding_mask = torch.cat([tgt_padding_mask, pad], dim=1)
            elif tgt_padding_mask.shape[1] > max_pos_len:
                tgt_padding_mask = tgt_padding_mask[:, :max_pos_len]
            # Transformer 前向傳播
            output = self.transformer(
                src_emb, tgt_emb,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            # 投影到 token 空間
            logits = self.output_projection(output)
            return logits
        
        else:
            # 推理模式：使用 encoder 進行 self-attention
            src_padding_mask = self.create_padding_mask(src_tokens_padded)
            
            # 僅使用 encoder
            encoder_output = self.transformer.encoder(
                src_emb,
                src_key_padding_mask=src_padding_mask
            )
            
            # 投影到 token 空間
            logits = self.output_projection(encoder_output)
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            return predicted_tokens
    
    def forward(self, noisy_audio, clean_audio=None):
        """完整的前向傳播：Audio → Tokens → Transformer → Tokens → Audio"""
        
        # Step 1: 將音頻轉換為 tokens (使用凍結的 WavTokenizer Encoder)
        noisy_tokens = self.encode_audio_to_tokens(noisy_audio)
        
        if self.training and clean_audio is not None:
            # 訓練模式：需要 clean tokens 作為目標
            clean_tokens = self.encode_audio_to_tokens(clean_audio)
            
            # 準備訓練所需的序列
            # 輸入序列：noisy tokens + EOS
            input_tokens = torch.cat([
                noisy_tokens, 
                torch.full((noisy_tokens.size(0), 1), self.eos_token, 
                          device=noisy_tokens.device, dtype=torch.long)
            ], dim=1)
            
            # 解碼器輸入：SOS + clean tokens (teacher forcing)
            decoder_input = torch.cat([
                torch.full((clean_tokens.size(0), 1), self.sos_token, 
                          device=clean_tokens.device, dtype=torch.long),
                clean_tokens
            ], dim=1)
            
            # 目標序列：clean tokens + EOS (與logits輸出長度匹配)
            target_tokens = torch.cat([
                clean_tokens,
                torch.full((clean_tokens.size(0), 1), self.eos_token, 
                          device=clean_tokens.device, dtype=torch.long)
            ], dim=1)
            
            # Step 2: Transformer 降噪 (在 token 空間)
            logits = self.forward_transformer(input_tokens, decoder_input)
            # 確保 target_tokens shape 與 logits 一致
            max_pos_len = logits.shape[1]
            if target_tokens.shape[1] < max_pos_len:
                pad_size = max_pos_len - target_tokens.shape[1]
                pad = torch.full((target_tokens.shape[0], pad_size), self.pad_token, device=target_tokens.device, dtype=target_tokens.dtype)
                target_tokens = torch.cat([target_tokens, pad], dim=1)
            elif target_tokens.shape[1] > max_pos_len:
                target_tokens = target_tokens[:, :max_pos_len]
            return {
                'logits': logits,
                'target_tokens': target_tokens,
                'noisy_tokens': noisy_tokens,
                'clean_tokens': clean_tokens
            }
        
        else:
            # 推理模式：直接降噪
            # Step 2: Transformer 降噪
            denoised_tokens = self.forward_transformer(noisy_tokens)
            
            # Step 3: 將 denoised tokens 轉換回音頻 (使用凍結的 WavTokenizer Decoder)
            denoised_audio = self.decode_tokens_to_audio(denoised_tokens)
            
            return {
                'denoised_audio': denoised_audio,
                'denoised_tokens': denoised_tokens,
                'noisy_tokens': noisy_tokens
            }

class AudioTokenDataset(Dataset):
    """音頻數據集，直接處理音頻而非預提取的 tokens"""
    
    def __init__(self, audio_dataset, max_audio_length=61920*3):  # 約3秒的音頻
        """
        Args:
            audio_dataset: AudioDataset 實例
            max_audio_length: 最大音頻長度（樣本點數）
        """
        self.audio_dataset = audio_dataset
        self.max_audio_length = max_audio_length
    
    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self, idx):
        # 從 AudioDataset 獲取音頻數據
        noisy_audio, clean_audio, content_id = self.audio_dataset[idx]
        
        # 確保音頻長度不超過限制
        if noisy_audio.size(-1) > self.max_audio_length:
            noisy_audio = noisy_audio[..., :self.max_audio_length]
        if clean_audio.size(-1) > self.max_audio_length:
            clean_audio = clean_audio[..., :self.max_audio_length]
        
        # 確保音頻維度正確 [1, length]
        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)
        
        return (noisy_audio, clean_audio, content_id)

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """使用原始 CrossEntropy 訓練一個 epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # batch 是 tuple 格式: (noisy_audio, clean_audio, content_id)
        noisy_audio = batch[0].to(device)
        clean_audio = batch[1].to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        output = model(noisy_audio, clean_audio)
        
        logits = output['logits']  # [batch_size, seq_len, vocab_size]
        target_tokens = output['target_tokens']  # [batch_size, seq_len]
        
        # 計算損失（忽略 padding tokens）- 使用reshape確保tensor連續性
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = target_tokens.reshape(-1)
        # 只對 codebook tokens 計算損失（忽略特殊 tokens）
        mask = target_flat < model.codebook_size
        if mask.sum() > 0:
            loss = criterion(logits_flat[mask], target_flat[mask])
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        # 反向傳播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 統計
        total_loss += loss.item()
        total_tokens += mask.sum().item()
        
        # 更新進度條
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Tokens': total_tokens})
    
    return total_loss / len(dataloader)

def train_epoch_with_token_loss(model, dataloader, optimizer, device, epoch, 
                               loss_weights={'l2': 0.3, 'consistency': 0.4, 'manifold': 0.1, 
                                           'normalization': 0.1, 'coherence': 0.1}):
    """使用 Token Loss 系統訓練一個 epoch（ttt2.py 損失邏輯移植到離散空間）"""
    model.train()
    total_losses = {'total': 0.0}
    loss_counts = 0
    
    # 獲取嵌入層用於 Token Loss 計算
    embedding_layer = None
    if hasattr(model, 'src_embedding'):
        embedding_layer = model.src_embedding
    elif hasattr(model, 'tgt_embedding'):
        embedding_layer = model.tgt_embedding
    
    if embedding_layer is None:
        logging.warning("無法找到嵌入層，將使用簡化的 token loss")
    else:
        logging.info(f"找到嵌入層：{type(embedding_layer).__name__}, 嵌入維度：{embedding_layer.embedding_dim}")
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (Token Loss)")
    
    for batch_idx, batch in enumerate(progress_bar):
        # batch 是 tuple 格式: (noisy_audio, clean_audio, content_id)
        noisy_audio = batch[0].to(device)
        clean_audio = batch[1].to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        output = model(noisy_audio, clean_audio)
        
        logits = output['logits']  # [batch_size, seq_len, vocab_size]
        target_tokens = output['target_tokens']  # [batch_size, seq_len]
        noisy_tokens = output['noisy_tokens']  # [batch_size, seq_len]
        
        # 獲取預測 tokens
        predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        
        # 使用 Token Loss 系統計算損失（ttt2.py 邏輯移植）
        try:
            total_loss, loss_dict = compute_combined_token_loss(
                predicted_logits=logits,
                predicted_tokens=predicted_tokens,
                target_tokens=target_tokens,
                input_tokens=noisy_tokens,
                embedding_layer=embedding_layer,
                weights=loss_weights  # 修正參數名稱
            )
        except Exception as e:
            logging.warning(f"Token loss 計算失敗，回退到交叉熵: {e}")
            # 回退到簡單交叉熵 - 使用reshape確保tensor連續性
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_tokens.reshape(-1)
            mask = target_flat < model.codebook_size
            if mask.sum() > 0:
                total_loss = F.cross_entropy(logits_flat[mask], target_flat[mask])
            else:
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict = {'total_loss': total_loss.item(), 'consistency_loss': total_loss.item()}
        
        # 反向傳播
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
    
    # 限制驗證批次數量以節省內存
    max_val_batches = 50
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch_count >= max_val_batches:
                break
                
            try:
                # batch 是 tuple 格式: (noisy_audio, clean_audio, content_id)
                noisy_audio = batch[0].to(device)
                clean_audio = batch[1].to(device)
                
                # 將音頻轉換為tokens
                with torch.no_grad():
                    # 確保音頻張量是正確的形狀 [B, 1, T]
                    while noisy_audio.dim() > 3:
                        noisy_audio = noisy_audio.squeeze()
                    while clean_audio.dim() > 3:
                        clean_audio = clean_audio.squeeze()
                    
                    # 確保是3D張量
                    if noisy_audio.dim() == 2:
                        noisy_audio = noisy_audio.unsqueeze(1)
                    if clean_audio.dim() == 2:
                        clean_audio = clean_audio.unsqueeze(1)
                        
                    # 編碼音頻為tokens
                    noisy_tokens = model.wavtokenizer.encode_infer(noisy_audio, bandwidth_id=torch.tensor([0]))
                    clean_tokens = model.wavtokenizer.encode_infer(clean_audio, bandwidth_id=torch.tensor([0]))
                    
                    # 提取第一層tokens [batch_size, time_frames] 並轉換為整數
                    noisy_tokens = noisy_tokens[0][0].squeeze(1).long()  # 轉為長整數
                    clean_tokens = clean_tokens[0][0].squeeze(1).long()  # 轉為長整數
                    
                    # 檢查token範圍並截斷超出詞彙的tokens
                    max_token = model.codebook_size - 1
                    noisy_tokens = torch.clamp(noisy_tokens, 0, max_token)
                    clean_tokens = torch.clamp(clean_tokens, 0, max_token)
                
                # 前向傳播 - 設置為訓練模式以獲得logits，但不計算梯度
                model.train()  # 臨時設為訓練模式
                
                # 準備訓練所需的序列格式
                input_tokens = torch.cat([
                    noisy_tokens, 
                    torch.full((noisy_tokens.size(0), 1), model.eos_token, 
                              device=noisy_tokens.device, dtype=torch.long)
                ], dim=1)
                
                decoder_input = torch.cat([
                    torch.full((clean_tokens.size(0), 1), model.sos_token, 
                              device=clean_tokens.device, dtype=torch.long),
                    clean_tokens
                ], dim=1)
                
                target_tokens = torch.cat([
                    clean_tokens,
                    torch.full((clean_tokens.size(0), 1), model.eos_token, 
                              device=clean_tokens.device, dtype=torch.long)
                ], dim=1)
                
                # 添加 padding 以匹配模型期望的序列長度（與訓練階段一致）
                max_pos_len = model.max_length  # 使用模型定義的最大長度
                current_seq_len = input_tokens.size(1)
                
                if current_seq_len < max_pos_len:
                    pad_len = max_pos_len - current_seq_len
                    
                    # 對 input_tokens 進行 padding
                    input_padding = torch.full((input_tokens.size(0), pad_len), model.pad_token,
                                             device=input_tokens.device, dtype=torch.long)
                    input_tokens = torch.cat([input_tokens, input_padding], dim=1)
                    
                    # 對 decoder_input 進行 padding
                    decoder_padding = torch.full((decoder_input.size(0), pad_len), model.pad_token,
                                                device=decoder_input.device, dtype=torch.long)
                    decoder_input = torch.cat([decoder_input, decoder_padding], dim=1)
                    
                    # 對 target_tokens 進行 padding
                    target_padding = torch.full((target_tokens.size(0), pad_len), model.pad_token,
                                               device=target_tokens.device, dtype=torch.long)
                    target_tokens = torch.cat([target_tokens, target_padding], dim=1)
                elif current_seq_len > max_pos_len:
                    # 如果序列太長，截斷到最大長度
                    input_tokens = input_tokens[:, :max_pos_len]
                    decoder_input = decoder_input[:, :max_pos_len]
                    target_tokens = target_tokens[:, :max_pos_len]
                
                # 使用transformer前向傳播
                logits = model.forward_transformer(input_tokens, decoder_input)
                
                model.eval()  # 恢復驗證模式
                
                # 計算損失 - 使用reshape而非view確保tensor連續性
                logits_flat = logits.reshape(-1, logits.size(-1))
                target_flat = target_tokens.reshape(-1)
                
                # 確保mask和target維度匹配
                mask = target_flat < model.codebook_size
                if mask.size(0) != logits_flat.size(0):
                    # 如果維度不匹配，截斷到較小的維度
                    min_size = min(mask.size(0), logits_flat.size(0))
                    mask = mask[:min_size]
                    logits_flat = logits_flat[:min_size]
                    target_flat = target_flat[:min_size]
                    logging.warning(f"驗證批次 {batch_idx}: 維度不匹配，已截斷到 {min_size}")
                
                if mask.sum() > 0:
                    loss = criterion(logits_flat[mask], target_flat[mask])
                    total_loss += loss.item()
                    
                    # 計算準確率
                    predictions = torch.argmax(logits_flat[mask], dim=-1)
                    total_correct += (predictions == target_flat[mask]).sum().item()
                    total_tokens += mask.sum().item()
                
                batch_count += 1
                
                # 清理內存
                del noisy_audio, clean_audio, noisy_tokens, clean_tokens
                del input_tokens, decoder_input, target_tokens, logits
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"驗證批次 {batch_count} 出錯，跳過: {e}")
                batch_count += 1
                continue
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return avg_loss, accuracy

def save_sample_ttt2_style(input_audio, output_audio, target_audio, epoch, batch_idx, save_dir, device):
    """保存音頻樣本和頻譜圖，完全比照 ttt2.py 的 save_sample 函數
    
    Args:
        input_audio: 重建的輸入音頻 (經過 WavTokenizer encode-decode)
        output_audio: 模型輸出音頻 (經過 Transformer + WavTokenizer 降噪)
        target_audio: 重建的目標音頻 (經過 WavTokenizer encode-decode)
        epoch: 當前訓練週期
        batch_idx: 批次索引
        save_dir: 保存目錄
        device: 計算設備
        
    重要修正:
        - input/target 現在都經過 WavTokenizer 重建，確保公平比較
        - enhanced 經過 Transformer + WavTokenizer，展現降噪效果
        - 三種音檔都具有相同的 WavTokenizer 基準品質 (~0.3-0.4 correlation)
    """
    import random
    
    try:
        # 調試：打印張量形狀
        logging.info(f"張量形狀 - input_audio: {input_audio.shape}, output_audio: {output_audio.shape}, target_audio: {target_audio.shape}")
        
        # 確保所有張量形狀一致 (處理4D -> 3D問題)
        if input_audio.dim() == 4:  # [B, C, 1, T] -> [B, C, T]
            input_audio = input_audio.squeeze(2)
            logging.info(f"調整input_audio形狀為: {input_audio.shape}")
            
        if output_audio.dim() == 4:  # [B, C, 1, T] -> [B, C, T]  
            output_audio = output_audio.squeeze(2)
            logging.info(f"調整output_audio形狀為: {output_audio.shape}")
            
        if target_audio.dim() == 4:  # [B, C, 1, T] -> [B, C, T]
            target_audio = target_audio.squeeze(2)
            logging.info(f"調整target_audio形狀為: {target_audio.shape}")
        
        # 建立音頻保存目錄 (使用固定的audio_samples目錄，與 ttt2.py 一致)
        audio_dir = os.path.join(save_dir, "audio_samples", f'epoch_{epoch+1}')
        os.makedirs(audio_dir, exist_ok=True)
        
        # 處理每個樣本 (與 ttt2.py 一致，最多處理3個樣本)
        batch_size = min(input_audio.size(0), 3)
        logging.info(f"將處理 {batch_size} 個樣本")
        
        for j in range(batch_size):  
            try:
                base_name = f"batch_{batch_idx}_sample_{j+1}"
                logging.info(f"處理樣本: {base_name}")
                
                with torch.no_grad():
                    # 按照 ttt2.py 的方式處理音頻：不做額外的flatten和reshape
                    # 只進行與 ttt2.py 相同的正規化
                    input_sample = input_audio[j:j+1]  # 保持 [1, C, T] 或 [1, T] 形狀
                    target_sample = target_audio[j:j+1]  # 保持原始形狀
                    output_sample = output_audio[j:j+1]  # 保持原始形狀
                    
                    # 使用與 ttt2.py 完全相同的正規化方式
                    input_sample = input_sample / (torch.max(torch.abs(input_sample)) + 1e-8)
                    target_sample = target_sample / (torch.max(torch.abs(target_sample)) + 1e-8)
                    output_sample = output_sample / (torch.max(torch.abs(output_sample)) + 1e-8)
                    
                    # 確保音頻是正確的2D形狀 [channels, time] 以便保存
                    if input_sample.dim() == 3:  # [1, 1, T] -> [1, T]
                        input_sample = input_sample.squeeze(1)
                    if target_sample.dim() == 3:  # [1, 1, T] -> [1, T]  
                        target_sample = target_sample.squeeze(1)
                    if output_sample.dim() == 3:  # [1, 1, T] -> [1, T]
                        output_sample = output_sample.squeeze(1)
                    
                    logging.info(f"樣本形狀 - input: {input_sample.shape}, output: {output_sample.shape}, target: {target_sample.shape}")
                    
                    # 保存每個音頻樣本和頻譜圖 (與 ttt2.py 一致)
                    for audio, prefix in [
                        (output_sample, "enhanced"),  # 對應 ttt2.py 的 enhanced
                        (input_sample, "input"),      # 對應 ttt2.py 的 input
                        (target_sample, "target")     # 對應 ttt2.py 的 target
                    ]:
                        # 保存音頻文件 (與 ttt2.py 一致)
                        audio_path = os.path.join(audio_dir, f"{base_name}_{prefix}.wav")
                        try:
                            # 嘗試使用與 ttt2.py 相同的 save_audio 函數
                            try:
                                from encoder.utils import save_audio
                                save_audio(audio.cpu(), audio_path, sample_rate=24000, rescale=True)
                                logging.info(f"🔊 保存{prefix}音頻到: {audio_path}")
                            except ImportError:
                                # 備用方案：使用torchaudio直接保存 (與 ttt2.py 的備用方案一致)
                                torchaudio.save(audio_path, audio.cpu(), 24000)
                                logging.info(f"🔊 使用torchaudio保存{prefix}音頻到: {audio_path}")
                        except Exception as save_err:
                            logging.warning(f"❌ 保存音頻失敗: {str(save_err)}")
                            # 最終備用方案：與 ttt2.py 的 except 分支一致
                            torchaudio.save(audio_path, audio.cpu(), 24000)
                            logging.info(f"🔊 使用備用方案保存{prefix}音頻到: {audio_path}")
                        
                        # 生成頻譜圖 (完全比照 ttt2.py)
                        spec_path = audio_path.replace(".wav", "_spec.png")
                        try:
                            plt.figure(figsize=(10, 4))
                            audio_np = audio.cpu().numpy().flatten()
                            
                            # 使用librosa生成頻譜圖 (與 ttt2.py 一致)
                            import librosa
                            import librosa.display
                            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
                            librosa.display.specshow(D, sr=24000, x_axis="time", y_axis="log")
                            plt.colorbar(format="%+2.0f dB")
                            plt.title(f"Epoch {epoch+1} {prefix.capitalize()} Spectrogram")                            
                            plt.tight_layout()
                            plt.savefig(spec_path)
                            plt.close()
                            logging.info(f"📊 保存{prefix}頻譜圖到: {spec_path}")
                        except Exception as spec_err:
                            logging.warning(f"❌ 生成頻譜圖時出錯: {str(spec_err)}")
                        
            except Exception as e:
                logging.warning(f"❌ 處理樣本 {j+1} 時出錯: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"❌ save_sample_ttt2_style函數錯誤: {str(e)}")
        return False
    
    return True


def save_audio_samples(model, dataloader, epoch, output_dir, device, num_samples=3):
    """保存音頻樣本用於質量檢查（與 ttt2.py 一致的音頻保存邏輯）
    
    Args:
        model: 模型實例
        dataloader: 資料載入器
        epoch: 當前epoch
        output_dir: 輸出目錄
        device: 計算設備
        num_samples: 保存樣本數量
    """
    import torchaudio
    
    model.eval()
    
    # 建立音頻保存目錄
    audio_dir = os.path.join(output_dir, "audio_samples", f'epoch_{epoch+1}')
    os.makedirs(audio_dir, exist_ok=True)
    
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if saved_count >= num_samples:
                break
                
            try:
                # batch 是 tuple 格式: (noisy_audio, clean_audio, content_id)
                noisy_audio = batch[0].to(device)
                clean_audio = batch[1].to(device)
                
                # 確保音頻張量是正確的形狀 [B, 1, T]
                while noisy_audio.dim() > 3:
                    noisy_audio = noisy_audio.squeeze()
                while clean_audio.dim() > 3:
                    clean_audio = clean_audio.squeeze()
                
                # 確保是3D張量
                if noisy_audio.dim() == 2:
                    noisy_audio = noisy_audio.unsqueeze(1)
                if clean_audio.dim() == 2:
                    clean_audio = clean_audio.unsqueeze(1)
                
                # 使用模型進行推理（inference mode）
                denoised_audio = model(noisy_audio)  # 使用單參數推理模式
                
                # 處理每個樣本
                for j in range(min(noisy_audio.size(0), num_samples - saved_count)):
                    try:
                        base_name = f"epoch_{epoch+1}_sample_{saved_count+1}"
                        
                        # 提取單個樣本
                        noisy_sample = noisy_audio[j].squeeze().cpu()
                        clean_sample = clean_audio[j].squeeze().cpu() 
                        denoised_sample = denoised_audio[j].squeeze().cpu()
                        
                        # 正規化音頻
                        noisy_sample = noisy_sample / (torch.max(torch.abs(noisy_sample)) + 1e-8)
                        clean_sample = clean_sample / (torch.max(torch.abs(clean_sample)) + 1e-8)
                        denoised_sample = denoised_sample / (torch.max(torch.abs(denoised_sample)) + 1e-8)
                        
                        # 保存音頻檔案
                        torchaudio.save(
                            os.path.join(audio_dir, f"{base_name}_noisy.wav"),
                            noisy_sample.unsqueeze(0), 24000
                        )
                        torchaudio.save(
                            os.path.join(audio_dir, f"{base_name}_clean.wav"),
                            clean_sample.unsqueeze(0), 24000
                        )
                        torchaudio.save(
                            os.path.join(audio_dir, f"{base_name}_denoised.wav"),
                            denoised_sample.unsqueeze(0), 24000
                        )
                        
                        saved_count += 1
                        logging.info(f"保存音頻樣本: {base_name}")
                        
                        if saved_count >= num_samples:
                            break
                            
                    except Exception as e:
                        logging.warning(f"保存樣本 {j} 失敗: {e}")
                        continue
                        
                if saved_count >= num_samples:
                    break
                    
            except Exception as e:
                logging.warning(f"處理批次 {batch_idx} 失敗: {e}")
                continue
    
    model.train()
    logging.info(f"完成音頻樣本保存，共保存 {saved_count} 個樣本")


def save_spectrograms(model, dataloader, epoch, output_dir, device, num_samples=3):
    """保存頻譜圖用於視覺化分析
    
    Args:
        model: 模型實例
        dataloader: 資料載入器
        epoch: 當前epoch
        output_dir: 輸出目錄
        device: 計算設備
        num_samples: 保存樣本數量
    """
    import matplotlib.pyplot as plt
    import torchaudio.transforms as T
    
    model.eval()
    
    # 建立頻譜圖保存目錄
    spec_dir = os.path.join(output_dir, "spectrograms", f'epoch_{epoch+1}')
    os.makedirs(spec_dir, exist_ok=True)
    
    # 設置頻譜圖轉換
    n_fft = 2048
    hop_length = 512
    spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if saved_count >= num_samples:
                break
                
            try:
                # batch 是 tuple 格式: (noisy_audio, clean_audio, content_id)
                noisy_audio = batch[0].to(device)
                clean_audio = batch[1].to(device)
                
                # 確保音頻張量是正確的形狀 [B, 1, T]
                while noisy_audio.dim() > 3:
                    noisy_audio = noisy_audio.squeeze()
                while clean_audio.dim() > 3:
                    clean_audio = clean_audio.squeeze()
                
                # 確保是3D張量
                if noisy_audio.dim() == 2:
                    noisy_audio = noisy_audio.unsqueeze(1)
                if clean_audio.dim() == 2:
                    clean_audio = clean_audio.unsqueeze(1)
                
                # 使用模型進行推理（inference mode）
                denoised_audio = model(noisy_audio)  # 使用單參數推理模式
                
                # 處理每個樣本
                for j in range(min(noisy_audio.size(0), num_samples - saved_count)):
                    try:
                        base_name = f"epoch_{epoch+1}_sample_{saved_count+1}"
                        
                        # 提取單個樣本並轉移到CPU
                        noisy_sample = noisy_audio[j].squeeze().cpu()
                        clean_sample = clean_audio[j].squeeze().cpu() 
                        denoised_sample = denoised_audio[j].squeeze().cpu()
                        
                        # 計算頻譜圖
                        noisy_spec = spec_transform(noisy_sample)
                        clean_spec = spec_transform(clean_sample)
                        denoised_spec = spec_transform(denoised_sample)
                        
                        # 轉換為dB尺度
                        noisy_spec_db = 10 * torch.log10(noisy_spec + 1e-8)
                        clean_spec_db = 10 * torch.log10(clean_spec + 1e-8)
                        denoised_spec_db = 10 * torch.log10(denoised_spec + 1e-8)
                        
                        # 創建對比圖
                        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
                        
                        # 噪聲音頻頻譜圖
                        im1 = axes[0].imshow(noisy_spec_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
                        axes[0].set_title(f'EXP-WAVTOK-CE-{epoch+1:04d} - 噪聲音頻頻譜圖 - 樣本 {saved_count+1}')
                        axes[0].set_ylabel('頻率 (bins)')
                        plt.colorbar(im1, ax=axes[0], label='功率 (dB)')
                        
                        # 清潔音頻頻譜圖
                        im2 = axes[1].imshow(clean_spec_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
                        axes[1].set_title(f'EXP-WAVTOK-CE-{epoch+1:04d} - 清潔音頻頻譜圖 - 樣本 {saved_count+1}')
                        axes[1].set_ylabel('頻率 (bins)')
                        plt.colorbar(im2, ax=axes[1], label='功率 (dB)')
                        
                        # 降噪音頻頻譜圖
                        im3 = axes[2].imshow(denoised_spec_db.numpy(), aspect='auto', origin='lower', cmap='viridis')
                        axes[2].set_title(f'EXP-WAVTOK-CE-{epoch+1:04d} - 降噪音頻頻譜圖 - 樣本 {saved_count+1}')
                        axes[2].set_xlabel('時間 (frames)')
                        axes[2].set_ylabel('頻率 (bins)')
                        plt.colorbar(im3, ax=axes[2], label='功率 (dB)')
                        
                        plt.tight_layout()
                        
                        # 保存頻譜圖
                        spec_path = os.path.join(spec_dir, f"{base_name}_spectrograms.png")
                        plt.savefig(spec_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        saved_count += 1
                        logging.info(f"保存頻譜圖: {base_name}")
                        
                        if saved_count >= num_samples:
                            break
                            
                    except Exception as e:
                        logging.warning(f"保存頻譜圖樣本 {j} 失敗: {e}")
                        continue
                        
                if saved_count >= num_samples:
                    break
                    
            except Exception as e:
                logging.warning(f"處理頻譜圖批次 {batch_idx} 失敗: {e}")
                continue
    
    model.train()
    logging.info(f"完成頻譜圖保存，共保存 {saved_count} 個樣本")


def save_checkpoint(model, optimizer, epoch, loss, save_path, config=None):
    """保存模型檢查點"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }
    torch.save(checkpoint, save_path)
    logging.info(f"檢查點已保存到: {save_path}")

def plot_training_history(train_losses, val_losses, val_accuracies, save_path):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 損失曲線
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 準確率曲線
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
    parser = argparse.ArgumentParser(description='WavTokenizer-Transformer 端到端音頻降噪')
    
    # WavTokenizer 參數
    parser.add_argument('--config', type=str, 
                        default='config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                        help='WavTokenizer 配置文件路徑')
    parser.add_argument('--model_path', type=str, 
                        default='models/wavtokenizer_large_speech_320_24k.ckpt',
                        help='WavTokenizer 預訓練模型路徑')
    
    # 模型參數 - 超輕量化配置以大幅減少記憶體使用和訓練時間
    parser.add_argument('--d_model', type=int, default=128, help='Transformer 模型維度')
    parser.add_argument('--nhead', type=int, default=2, help='注意力頭數')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='編碼器層數')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='解碼器層數')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='前饋網絡維度')
    parser.add_argument('--max_length', type=int, default=256, help='最大序列長度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 率')
    
    # 訓練參數 - 記憶體友好設定
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='權重衰減')
    parser.add_argument('--save_every', type=int, default=10, help='每幾個 epoch 保存一次')
    
    # 記憶體優化參數
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累積步數')
    parser.add_argument('--mixed_precision', action='store_true', help='啟用混合精度訓練')
    parser.add_argument('--device', type=str, default='cuda', help='訓練設備')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader 工作線程數')
    parser.add_argument('--pin_memory', action='store_true', help='啟用pin_memory')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='預取因子')
    parser.add_argument('--max_audio_length', type=float, default=3.0, help='最大音頻長度(秒)')
    parser.add_argument('--save_interval', type=int, default=50, help='保存間隔')
    parser.add_argument('--log_interval', type=int, default=10, help='日誌間隔')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='預熱步數')
    parser.add_argument('--use_scheduler', action='store_true', help='使用學習率調度器')
    parser.add_argument('--token_loss_weight', type=float, default=1.0, help='Token Loss權重')
    
    # 損失函數選擇
    parser.add_argument('--use_token_loss', action='store_true', 
                        help='使用 Token Loss 系統（ttt2.py 邏輯移植到離散空間）而非單純交叉熵')
    parser.add_argument('--l2_weight', type=float, default=0.3, help='L2 距離損失權重')
    parser.add_argument('--consistency_weight', type=float, default=0.4, help='內容一致性損失權重')
    parser.add_argument('--manifold_weight', type=float, default=0.1, help='Manifold 正則化權重')
    parser.add_argument('--normalization_weight', type=float, default=0.1, help='正則化損失權重')
    parser.add_argument('--coherence_weight', type=float, default=0.1, help='連貫性損失權重')

    # 數據參數
    parser.add_argument('--output_dir', type=str, default='results/wavtokenizer_transformer_denoising',
                        help='輸出目錄')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大處理樣本數 (None 表示使用全部數據)')
    parser.add_argument('--max_sentences_per_speaker', type=int, default=100,
                        help='每位語者最大句子數')
    parser.add_argument('--val_speakers', nargs='+', default=['girl9', 'boy7'],
                        help='驗證集語者')
    parser.add_argument('--train_speakers', nargs='+', default=None,
                        help='訓練集語者 (如果指定，只使用這些語者進行訓練；如果不指定，使用除驗證集外的所有語者)')
    
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
    
    # 設定設備 - 若已設 CUDA_VISIBLE_DEVICES，則只會看到一張 GPU，gpu_id 應為 0
    if torch.cuda.is_available():
        # 若只看到一張 GPU（如 CUDA_VISIBLE_DEVICES=2），gpu_id 必須設為 0
        visible_gpus = torch.cuda.device_count()
        if visible_gpus == 1:
            gpu_id = 0
        else:
            # 預設仍選空閒的 GPU 2（僅在未設 CUDA_VISIBLE_DEVICES 時有效）
            gpu_id = 2
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        # 設置內存優化
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False  # 節省內存
        torch.cuda.set_per_process_memory_fraction(0.8, device=gpu_id)  # 可以使用更多內存
        
        logging.info(f"使用設備: {device} (GPU 2 完全空閒，可以使用更多內存)")
    else:
        device = torch.device("cpu")
        logging.info(f"使用設備: {device}")
    
    # 準備數據集
    logging.info("準備音頻資料集...")
    
    # 使用與 ttt2.py 相同的數據配置
    input_dirs = [os.path.join(os.getcwd(), "data", "raw", "box")]
    target_dir = os.path.join(os.getcwd(), "data", "clean", "box2")
    
    if not any(os.path.exists(d) for d in input_dirs) or not os.path.exists(target_dir):
        logging.error("未找到數據目錄，請確保數據路徑正確")
        return
    
    # 創建音頻數據集
    audio_dataset = AudioDataset(input_dirs, target_dir, 
                                max_sentences_per_speaker=args.max_sentences_per_speaker)
    
    logging.info(f"數據集大小: {len(audio_dataset)} 個音頻對")
    
    # 按語者分割數據集（符合實驗設計：可指定訓練和驗證語者）
    logging.info(f"按語者分割數據集，驗證集語者: {args.val_speakers}")
    if args.train_speakers:
        logging.info(f"指定訓練集語者: {args.train_speakers}")
    else:
        logging.info("訓練集語者: 除驗證集外的所有語者")
    
    train_indices = []
    val_indices = []
    
    # 遍歷每個樣本，根據語者分配到訓練或驗證集
    for idx in range(len(audio_dataset)):
        audio_data = audio_dataset.paired_files[idx]  # 獲取原始音頻數據
        speaker = audio_data['speaker']
        
        if speaker in args.val_speakers:
            val_indices.append(idx)
        elif args.train_speakers is None or speaker in args.train_speakers:
            # 如果沒有指定訓練語者，使用除驗證集外的所有語者
            # 如果指定了訓練語者，只使用指定的語者
            train_indices.append(idx)
        # 如果語者既不在驗證集也不在指定的訓練集中，則跳過
    
    # 限制樣本數（用於快速測試）
    if args.max_samples and args.max_samples < len(audio_dataset):
        # 計算每個集合應該有多少樣本（保持相同比例）
        total_samples = len(train_indices) + len(val_indices)
        train_ratio = len(train_indices) / total_samples if total_samples > 0 else 0.8
        val_ratio = len(val_indices) / total_samples if total_samples > 0 else 0.2
        
        max_train_samples = max(1, int(args.max_samples * train_ratio))
        max_val_samples = max(1, args.max_samples - max_train_samples)
        
        # 限制每個集合的樣本數
        train_indices = train_indices[:max_train_samples]
        val_indices = val_indices[:max_val_samples]
        
        logging.info(f"限制樣本數為: {args.max_samples} (訓練集: {len(train_indices)}, 驗證集: {len(val_indices)})")
    
    # 創建音頻-token 數據集
    dataset = AudioTokenDataset(audio_dataset)
    
    # 創建按語者分割的數據集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 創建數據載入器，使用 ttt2.py 的 collate_fn 處理不同長度的音訊
    train_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
    val_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=val_collate_fn)
    
    logging.info(f"訓練集大小: {len(train_dataset)}, 驗證集大小: {len(val_dataset)}")
    
    # 創建模型
    logging.info("創建 WavTokenizer-Transformer 降噪模型...")
    model = WavTokenizerTransformerDenoiser(
        config_path=args.config,
        model_path=args.model_path,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_length=args.max_length,
        dropout=args.dropout
    ).to(device)
    
    # 計算模型參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"模型總參數數量: {total_params:,}")
    logging.info(f"可訓練參數數量: {trainable_params:,}")
    logging.info(f"WavTokenizer 參數數量 (凍結): {total_params - trainable_params:,}")
    
    # 創建優化器和損失函數 - GPU 2有充足內存，可以使用Adam
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
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
        try:
            # 選擇訓練函數
            if args.use_token_loss:
                # 使用 Token Loss 系統（ttt2.py 邏輯移植到離散空間）
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
            
            # 驗證
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
            
        except Exception as epoch_error:
            logging.error(f"Epoch {epoch} 出現錯誤: {epoch_error}")
            logging.error(f"錯誤類型: {type(epoch_error).__name__}")
            
            # 如果是關鍵錯誤，保存當前模型並繼續
            try:
                emergency_save_path = os.path.join(args.output_dir, f'emergency_checkpoint_epoch_{epoch}.pth')
                config = {
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_encoder_layers': args.num_encoder_layers,
                    'num_decoder_layers': args.num_decoder_layers,
                    'dim_feedforward': args.dim_feedforward,
                    'max_length': args.max_length,
                    'dropout': args.dropout,
                    'config_path': args.config,
                    'model_path': args.model_path
                }
                save_checkpoint(model, optimizer, epoch-1, float('inf'), emergency_save_path, config)
                logging.info(f"緊急檢查點已保存到: {emergency_save_path}")
            except Exception as save_error:
                logging.error(f"緊急保存失敗: {save_error}")
            
            # 添加假值以保持列表一致性
            if len(train_losses) < epoch:
                train_losses.append(float('inf'))
            if len(val_losses) < epoch:
                val_losses.append(float('inf'))
                val_accuracies.append(0.0)
                
            continue  # 繼續下一個epoch
        
        # 保存樣本和頻譜圖 (使用 save_every 參數控制頻率)
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            logging.info(f"Saving samples for epoch {epoch+1}...")
            try:
                model.eval()
                with torch.no_grad():
                    # 基於當前 epoch 計算偏移量，確保每次採樣到不同的批次 (與 ttt2.py 一致)
                    epoch_offset = (epoch // 100) % 10  # 每10次採樣循環一次
                    
                    # 創建一個臨時數據載入器，使用原始的train_loader的dataset (與 ttt2.py 一致)
                    dataset_to_use = train_loader.dataset
                    temp_loader = DataLoader(
                        dataset_to_use,
                        batch_size=args.batch_size,
                        shuffle=True,  # 強制隨機打亂
                        num_workers=1,
                        collate_fn=train_collate_fn,  # 使用原始的collate_fn函數
                        worker_init_fn=lambda _: random.seed(42 + epoch)  # 基於 epoch 設置隨機種子
                    )
                    
                    # 跳過前面的批次，實現每次採樣不同的數據 (與 ttt2.py 一致)
                    for _ in range(epoch_offset):
                        try:
                            next(iter(temp_loader))
                        except StopIteration:
                            break
                    
                    # 採樣並保存 (與 ttt2.py 一致)
                    for batch_idx, batch_data in enumerate(temp_loader):
                        # 解包數據，同時兼容包含content_id的數據格式 (與 ttt2.py 一致)
                        if len(batch_data) == 3:  # 包含content_id
                            input_wav, target_wav, _ = batch_data
                        else:  # 不包含content_id
                            input_wav, target_wav = batch_data
                        
                        input_wav = input_wav.to(device)
                        target_wav = target_wav.to(device)
                        
                        # 推理模式：使用模型生成降噪音頻 (比照 ttt2.py 的 output_tuple)
                        with torch.no_grad():
                            model_output = model(input_wav)  # 這會返回字典
                            if isinstance(model_output, dict) and 'denoised_audio' in model_output:
                                output_audio = model_output['denoised_audio']
                            else:
                                output_audio = model_output  # 備用方案
                            
                            # 🔧 修正：為了真實反映 WavTokenizer 系統性能
                            # input 和 target 也應該經過 WavTokenizer encode-decode 重建
                            # 這樣可以公平比較三種音檔的品質
                            
                            # 重建 input 音檔 (通過 WavTokenizer encode-decode)
                            input_tokens = model.encode_audio_to_tokens(input_wav)
                            input_reconstructed = model.decode_tokens_to_audio(input_tokens)
                            
                            # 重建 target 音檔 (通過 WavTokenizer encode-decode)  
                            target_tokens = model.encode_audio_to_tokens(target_wav)
                            target_reconstructed = model.decode_tokens_to_audio(target_tokens)
                        
                        # 使用與 ttt2.py 完全相同的 save_sample 邏輯
                        # 🔧 修正：使用經過 WavTokenizer 重建的音檔，確保公平比較
                        save_sample_ttt2_style(
                            input_audio=input_reconstructed,  # 經過 WavTokenizer 重建的 input
                            output_audio=output_audio,        # 經過 Transformer + WavTokenizer 的 enhanced
                            target_audio=target_reconstructed, # 經過 WavTokenizer 重建的 target
                            epoch=epoch,
                            batch_idx=batch_idx,
                            save_dir=args.output_dir,
                            device=device
                        )
                        
                        # 只保存前幾個 batch 的樣本 (與 ttt2.py 一致)
                        if batch_idx >= 2:  # 只保存前2個batch
                            break
                        
                model.train()
            except Exception as e:
                logging.warning(f"Failed to save sample for epoch {epoch+1}: {e}")
        
        # 繪製學習曲線 (與 ttt2.py 一致：每 50 epochs)
        if (epoch + 1) % 50 == 0:
            plot_training_history(
                train_losses, val_losses, val_accuracies,
                os.path.join(args.output_dir, f'training_history_epoch_{epoch+1}.png')
            )
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            config = {
                'd_model': args.d_model,
                'nhead': args.nhead,
                'num_encoder_layers': args.num_encoder_layers,
                'num_decoder_layers': args.num_decoder_layers,
                'dim_feedforward': args.dim_feedforward,
                'max_length': args.max_length,
                'dropout': args.dropout,
                'config_path': args.config,
                'model_path': args.model_path
            }
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, 'best_model.pth'),
                config
            )
            logging.info(f"新的最佳模型已保存 (Val Loss: {val_loss:.4f})")
        
        # 定期保存檢查點和音頻樣本 (每 100 epochs 或最後一個 epoch)
        if (epoch + 1) % 100 == 0 or epoch == args.num_epochs - 1:
            config = {
                'd_model': args.d_model,
                'nhead': args.nhead,
                'num_encoder_layers': args.num_encoder_layers,
                'num_decoder_layers': args.num_decoder_layers,
                'dim_feedforward': args.dim_feedforward,
                'max_length': args.max_length,
                'dropout': args.dropout,
                'config_path': args.config,
                'model_path': args.model_path
            }
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                config
            )
            logging.info(f"定期檢查點已保存 (Epoch {epoch+1})")
            
            # 注意：音頻樣本和頻譜圖保存已經在上面的 ttt2.py 風格保存中處理了
            # 舊的保存方式已移除，現在使用與 ttt2.py 完全一致的保存邏輯
        
        # 每 50 epochs 保存模型檢查點 (與 ttt2.py save_every 邏輯一致)
        if (epoch + 1) % args.save_every == 0:
            config = {
                'd_model': args.d_model,
                'nhead': args.nhead,
                'num_encoder_layers': args.num_encoder_layers,
                'num_decoder_layers': args.num_decoder_layers,
                'dim_feedforward': args.dim_feedforward,
                'max_length': args.max_length,
                'dropout': args.dropout,
                'config_path': args.config,
                'model_path': args.model_path
            }
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'),
                config
            )
    
    logging.info("訓練完成！")
    
    # 最終保存
    final_config = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'max_length': args.max_length,
        'dropout': args.dropout,
        'config_path': args.config,
        'model_path': args.model_path
    }
    save_checkpoint(
        model, optimizer, args.num_epochs, val_losses[-1],
        os.path.join(args.output_dir, 'final_model.pth'),
        final_config
    )
    
    plot_training_history(
        train_losses, val_losses, val_accuracies,
        os.path.join(args.output_dir, 'final_training_history.png')
    )

if __name__ == "__main__":
    main()

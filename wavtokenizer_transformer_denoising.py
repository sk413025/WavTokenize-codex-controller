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
                 d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, max_length=512, dropout=0.1):
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
            
            return tokens.long()
    
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
        
        # 源序列嵌入和位置編碼
        src_emb = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoding[:, :src_seq_len, :]
        
        # 如果是訓練模式且提供了目標序列
        if tgt_tokens is not None and self.training:
            _, tgt_seq_len = tgt_tokens.size()
            
            # 目標序列嵌入和位置編碼
            tgt_emb = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
            tgt_emb = tgt_emb + self.pos_encoding[:, :tgt_seq_len, :]
            
            # 創建遮罩
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src_tokens.device)
            src_padding_mask = self.create_padding_mask(src_tokens)
            tgt_padding_mask = self.create_padding_mask(tgt_tokens)
            
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
            src_padding_mask = self.create_padding_mask(src_tokens)
            
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
            
            # 目標序列：SOS + clean tokens + EOS
            target_tokens = torch.cat([
                torch.full((clean_tokens.size(0), 1), self.sos_token, 
                          device=clean_tokens.device, dtype=torch.long),
                clean_tokens,
                torch.full((clean_tokens.size(0), 1), self.eos_token, 
                          device=clean_tokens.device, dtype=torch.long)
            ], dim=1)
            
            # 解碼器輸入：SOS + clean tokens (teacher forcing)
            decoder_input = torch.cat([
                torch.full((clean_tokens.size(0), 1), self.sos_token, 
                          device=clean_tokens.device, dtype=torch.long),
                clean_tokens
            ], dim=1)
            
            # Step 2: Transformer 降噪 (在 token 空間)
            logits = self.forward_transformer(input_tokens, decoder_input)
            
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
        
        return {
            'noisy_audio': noisy_audio,
            'clean_audio': clean_audio,
            'content_id': content_id
        }

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """使用原始 CrossEntropy 訓練一個 epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        output = model(noisy_audio, clean_audio)
        
        logits = output['logits']  # [batch_size, seq_len, vocab_size]
        target_tokens = output['target_tokens']  # [batch_size, seq_len]
        
        # 計算損失（忽略 padding tokens）
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target_tokens.view(-1)
        
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
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        
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
                weights=loss_weights
            )
        except Exception as e:
            logging.warning(f"Token loss 計算失敗，回退到交叉熵: {e}")
            # 回退到簡單交叉熵
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_tokens.view(-1)
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
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            noisy_audio = batch['noisy_audio'].to(device)
            clean_audio = batch['clean_audio'].to(device)
            
            # 前向傳播
            output = model(noisy_audio, clean_audio)
            
            logits = output['logits']
            target_tokens = output['target_tokens']
            
            # 計算損失
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_tokens.view(-1)
            
            mask = target_flat < model.codebook_size
            if mask.sum() > 0:
                loss = criterion(logits_flat[mask], target_flat[mask])
                total_loss += loss.item()
                
                # 計算準確率
                predictions = torch.argmax(logits_flat[mask], dim=-1)
                total_correct += (predictions == target_flat[mask]).sum().item()
                total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return avg_loss, accuracy

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
    
    # 模型參數
    parser.add_argument('--d_model', type=int, default=512, help='Transformer 模型維度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力頭數')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='編碼器層數')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='解碼器層數')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='前饋網絡維度')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列長度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 率')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='權重衰減')
    parser.add_argument('--save_every', type=int, default=10, help='每幾個 epoch 保存一次')
    
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
    
    # 限制樣本數（用於快速測試）
    if args.max_samples and args.max_samples < len(audio_dataset):
        indices = list(range(args.max_samples))
        audio_dataset = torch.utils.data.Subset(audio_dataset, indices)
        logging.info(f"限制樣本數為: {args.max_samples}")
    
    # 創建音頻-token 數據集
    dataset = AudioTokenDataset(audio_dataset)
    
    # 按語者分割數據集（符合實驗設計：10人訓練、2人驗證）
    logging.info(f"按語者分割數據集，驗證集語者: {args.val_speakers}")
    
    train_indices = []
    val_indices = []
    
    # 遍歷每個樣本，根據語者分配到訓練或驗證集
    for idx in range(len(dataset)):
        audio_data = audio_dataset.paired_files[idx]  # 獲取原始音頻數據
        speaker = audio_data['speaker']
        
        if speaker in args.val_speakers:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    
    # 創建按語者分割的數據集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 創建數據載入器，使用 ttt2.py 的 collate_fn 處理不同長度的音訊
    train_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
    val_collate_fn = lambda batch: collate_fn(batch, trim_to_shortest=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=val_collate_fn)
    
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
    
    # 創建優化器和損失函數
    optimizer = optim.AdamW(
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
        
        # 保存樣本和頻譜圖 (與 ttt2.py 一致：每 100 epochs)
        if epoch % 100 == 0 or epoch == args.num_epochs:
            logging.info(f"Saving audio samples and spectrograms for epoch {epoch}...")
            try:
                model.eval()
                with torch.no_grad():
                    # 從驗證集取一個批次進行示例
                    for batch_idx, batch in enumerate(val_loader):
                        if batch_idx >= 1:  # 只處理第一個批次
                            break
                            
                        noisy_audio = batch['noisy_audio'][:1].to(device)  # 只取第一個樣本
                        clean_audio = batch['clean_audio'][:1].to(device)
                        
                        # 前向傳播獲取輸出
                        output = model(noisy_audio, clean_audio)
                        denoised_audio = output['output_audio']
                        
                        # 保存樣本和頻譜圖
                        save_sample_with_spectrograms(
                            input_audio=noisy_audio[0],
                            output_audio=denoised_audio[0], 
                            target_audio=clean_audio[0],
                            epoch=epoch,
                            batch_idx=0,
                            save_dir=args.output_dir,
                            prefix="validation"
                        )
                        break
                        
                model.train()
            except Exception as e:
                logging.warning(f"Failed to save sample for epoch {epoch}: {e}")
        
        # 繪製學習曲線 (與 ttt2.py 一致：每 50 epochs)
        if epoch % 50 == 0:
            plot_training_history(
                train_losses, val_losses, val_accuracies,
                os.path.join(args.output_dir, f'training_history_epoch_{epoch}.png')
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
        
        # 定期保存檢查點 (與 ttt2.py 一致：每 300 epochs 或最後一個 epoch)
        if epoch % 300 == 0 or epoch == args.num_epochs:
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
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'),
                config
            )
            logging.info(f"定期檢查點已保存 (Epoch {epoch})")
        
        # 每 50 epochs 保存模型檢查點 (與 ttt2.py save_every 邏輯一致)
        if epoch % args.save_every == 0:
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
                os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'),
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

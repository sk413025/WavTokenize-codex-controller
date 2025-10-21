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

# 設置 logger
logger = logging.getLogger(__name__)

class EmbeddingWrapper:
    """
    包裝器：讓模型的 _embed_tokens 方法可以像 nn.Embedding 一樣被調用
    
    這個包裝器允許 token_loss_system 使用模型的完整 embedding 流程，
    包括 codebook_embedding、special_token_embedding 和 projection。
    """
    def __init__(self, embed_func):
        self.embed_func = embed_func
    
    def __call__(self, tokens):
        return self.embed_func(tokens)

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
    """保存音頻樣本
    
    修復維度問題，確保音頻張量格式正確
    """
    try:
        # 確保音頻格式正確 - 更強健的維度處理
        if isinstance(audio, torch.Tensor):
            # 移除所有大小為1的維度
            audio = audio.squeeze()
            
            # 確保至少是2D張量 [channels, time]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # [time] -> [1, time]
            elif audio.dim() > 2:
                # 如果有多個維度，保留最後兩個維度
                audio = audio.view(-1, audio.size(-1))
                if audio.size(0) > 1:
                    # 如果有多個channel，取第一個
                    audio = audio[0:1]
        
        # 正規化音頻
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
        
        torchaudio.save(save_path, audio.cpu(), sample_rate)
    except Exception as e:
        print(f"Error saving audio: {str(e)}")

def normalize_audio_dimensions(audio):
    """標準化音頻張量維度為 [batch, 1, time]
    
    修復SConv1d期望3D但收到4D的問題
    """
    if not isinstance(audio, torch.Tensor):
        return audio
    
    # 移除所有大小為1的維度，除了批次維度
    while audio.dim() > 3:
        audio = audio.squeeze(-2) if audio.size(-2) == 1 else audio.squeeze(-1) if audio.size(-1) == 1 else audio.squeeze(1) if audio.size(1) == 1 and audio.dim() > 3 else audio.view(audio.size(0), -1, audio.size(-1))
        break
    
    # 確保是正確的3D格式 [batch, channels, time]
    if audio.dim() == 1:
        # [time] -> [1, 1, time]
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        # [batch, time] -> [batch, 1, time]
        audio = audio.unsqueeze(1)
    elif audio.dim() > 3:
        # 如果維度過多，reshape到正確格式
        batch_size = audio.size(0)
        audio = audio.view(batch_size, 1, -1)
    
    return audio

def apply_advanced_gradient_clipping(model, max_norm=0.5, adaptive=True):
    """改進的梯度裁剪策略
    
    解決梯度退化92.3%的問題
    主要改進：
    1. 自適應梯度裁剪閾值
    2. 梯度退化檢測
    3. 更保守的裁剪策略
    """
    # 計算梯度範數
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    if adaptive and param_count > 0:
        # 自適應調整裁剪閾值
        avg_norm = total_norm / param_count
        if avg_norm < 1e-6:
            # 梯度過小，說明可能有退化
            max_norm = min(max_norm * 2.0, 1.0)  # 放寬限制
        elif avg_norm > 10.0:
            # 梯度過大，需要更嚴格裁剪
            max_norm = max(max_norm * 0.5, 0.1)  # 更嚴格裁剪
    
    # 應用梯度裁剪
    if total_norm > max_norm:
        clip_ratio = max_norm / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_ratio)
        
        return max_norm
    
    return total_norm

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
        
        # 提取 WavTokenizer 的預訓練 codebook
        logging.info("提取 WavTokenizer 的 codebook 作為 token embedding...")
        pretrained_embeddings = self._extract_codebook_embeddings()
        codebook_dim = pretrained_embeddings.shape[1]
        
        # 創建兩個獨立的 Embedding 層
        # 1. codebook_embedding: 用於聲學 token (0-4095)，使用預訓練權重並凍結
        self.codebook_embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, 
            freeze=True  # 凍結預訓練權重
        )
        logging.info(f"成功創建 codebook embedding: shape={pretrained_embeddings.shape}, freeze=True")
        
        # 2. special_token_embedding: 用於 PAD, SOS, EOS (4096-4098)，可學習
        self.special_token_embedding = nn.Embedding(3, codebook_dim)
        logging.info(f"成功創建 special token embedding: 3 tokens, dim={codebook_dim}")
        
        # 3. 投影層：將 embedding 維度對齊到 Transformer 的 d_model
        if codebook_dim != d_model:
            self.embedding_projection = nn.Linear(codebook_dim, d_model)
            logging.info(f"Codebook 維度投影: {codebook_dim} → {d_model}")
        else:
            self.embedding_projection = nn.Identity()
            logging.info(f"Codebook 維度匹配: {codebook_dim}")
        
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
        # 🔴 修復: 使用 vocab_size (4099) 而非 codebook_size (4096)
        # 必須包含特殊 token (PAD=4096, SOS=4097, EOS=4098)
        self.output_projection = nn.Linear(d_model, self.vocab_size)
        
        # 初始化參數
        self._init_parameters()
    
    def _extract_codebook_embeddings(self):
        """
        從 WavTokenizer 提取預訓練的 codebook embeddings
        返回可以直接用於 nn.Embedding.from_pretrained() 的張量
        """
        try:
            vq_layers = self.wavtokenizer.feature_extractor.encodec.quantizer.vq.layers
            
            # 從第一個 VQ 層提取 codebook
            # 注意：如果模型有多層量化器，這裡只用第一層
            if len(vq_layers) == 1:
                codebook_embeddings = vq_layers[0].codebook
            else:
                # 如果有多層，拼接所有層的 codebook
                codebook_embeddings = torch.cat([vq.codebook for vq in vq_layers], dim=0)
            
            logging.info(f"成功提取 codebook embeddings: shape={codebook_embeddings.shape}")
            logging.info(f"- 來自 {len(vq_layers)} 個 VQ 層")
            logging.info(f"- 總共 {codebook_embeddings.shape[0]} 個 codes")
            logging.info(f"- 每個 code 維度: {codebook_embeddings.shape[1]}")
            
            # detach 以避免梯度計算（nn.Embedding.from_pretrained 會再處理）
            return codebook_embeddings.detach()
            
        except Exception as e:
            logging.error(f"提取 codebook embeddings 失敗: {e}")
            logging.error("請檢查 WavTokenizer 模型結構")
            raise
    
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
            # 確保tokens在詞彙範圍內 - 修復索引越界問題
            tokens = torch.clamp(tokens, 0, self.codebook_size - 1)
            
            # 額外檢查：確保沒有無效值
            if torch.any(tokens < 0) or torch.any(tokens >= self.vocab_size):
                print(f"Warning: 檢測到無效token值，範圍: {tokens.min().item()} to {tokens.max().item()}")
                tokens = torch.clamp(tokens, 0, self.codebook_size - 1)
            
            return tokens
    
    def decode_tokens_to_audio(self, tokens):
        """使用 WavTokenizer Decoder 將 tokens 轉換為音頻
        
        修復維度問題，確保輸出標準化的音頻格式
        """
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
            
            # 使用標準化函數修復維度問題
            audio = normalize_audio_dimensions(audio)
            
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
    
    def get_token_embeddings(self, tokens):
        """
        高效地獲取 token embeddings，使用兩個獨立的 Embedding 層：
        - codebook_embedding: 用於聲學 tokens (0-4095)，預訓練且凍結
        - special_token_embedding: 用於特殊 tokens (4096-4098)，可學習
        
        Args:
            tokens: [batch_size, seq_len] - token indices
                - 0 ~ 4095: codebook tokens
                - 4096: pad_token
                - 4097: sos_token  
                - 4098: eos_token
            
        Returns:
            embeddings: [batch_size, seq_len, d_model] - embedded representations
        """
        device = tokens.device
        codebook_dim = self.codebook_embedding.embedding_dim
        
        # 創建未投影的 embedding 張量（使用 codebook 的原始維度 512）
        raw_embeddings = torch.zeros(
            tokens.shape[0], tokens.shape[1], codebook_dim,
            device=device
        )
        
        # 創建 mask 來區分兩種類型的 token
        codebook_mask = tokens < self.codebook_size  # 0-4095
        special_mask = tokens >= self.codebook_size  # 4096-4098
        
        # 1. 處理 codebook tokens (0-4095)
        #    使用預訓練的 codebook_embedding（凍結）
        if codebook_mask.any():
            codebook_indices = tokens[codebook_mask]
            raw_embeddings[codebook_mask] = self.codebook_embedding(codebook_indices)
        
        # 2. 處理 special tokens (4096-4098)
        #    將 token ID 映射到 embedding 索引 (0, 1, 2)
        #    使用可學習的 special_token_embedding
        if special_mask.any():
            special_indices = tokens[special_mask] - self.codebook_size  # 映射: 4096→0, 4097→1, 4098→2
            raw_embeddings[special_mask] = self.special_token_embedding(special_indices)
        
        # 3. 統一投影到 Transformer 的 d_model 維度
        embeddings = self.embedding_projection(raw_embeddings)  # [B, L, 512] → [B, L, d_model]
        
        return embeddings
    
    def forward_transformer(self, src_tokens, tgt_tokens=None, return_logits=False):
        """Transformer 前向傳播（僅處理 token 序列）
        
        Args:
            src_tokens: 源 token 序列 [B, L]
            tgt_tokens: 目標 token 序列 [B, L]（訓練/驗證時提供）
            return_logits: 強制返回 logits 而非 predicted_tokens（用於驗證）
        
        Returns:
            training 模式或 return_logits=True: logits [B, L, vocab_size]
            inference 模式: predicted_tokens [B, L]
        """
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
            
        # 使用預訓練 codebook 計算 embedding
        src_emb = self.get_token_embeddings(src_tokens_padded) * math.sqrt(self.d_model)
        src_seq_len = src_tokens_padded.shape[1]
        src_emb = src_emb + self.pos_encoding[:, :src_seq_len, :]
        
        # 修改條件：訓練模式 OR 需要返回 logits（用於驗證）
        if tgt_tokens is not None and (self.training or return_logits):
            _, tgt_seq_len = tgt_tokens.size()
            max_pos_len = self.pos_encoding.shape[1]
            
            # 修正：先對 token ID 序列進行填充或裁剪，然後再做 embedding
            if tgt_seq_len < max_pos_len:
                pad_size = max_pos_len - tgt_seq_len
                pad = torch.full((tgt_tokens.shape[0], pad_size), self.pad_token, 
                               device=tgt_tokens.device, dtype=torch.long)
                tgt_tokens_padded = torch.cat([tgt_tokens, pad], dim=1)
            elif tgt_seq_len > max_pos_len:
                tgt_tokens_padded = tgt_tokens[:, :max_pos_len]
            else:
                tgt_tokens_padded = tgt_tokens
            
            # 對填充後的完整序列進行 embedding
            tgt_emb = self.get_token_embeddings(tgt_tokens_padded) * math.sqrt(self.d_model)
            actual_tgt_len = tgt_tokens_padded.shape[1]
            tgt_emb = tgt_emb + self.pos_encoding[:, :actual_tgt_len, :]
            
            # 創建遮罩（使用填充後的序列）
            tgt_mask = self.generate_square_subsequent_mask(actual_tgt_len).to(src_tokens.device)
            src_padding_mask = self.create_padding_mask(src_tokens_padded)
            tgt_padding_mask = self.create_padding_mask(tgt_tokens_padded)
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
            # 推理模式：快速 encoder-only 推理
            # 適用於去噪任務（輸入輸出長度相同）
            src_padding_mask = self.create_padding_mask(src_tokens_padded)
            
            # Encoder 處理輸入序列
            encoder_output = self.transformer.encoder(
                src_emb,
                src_key_padding_mask=src_padding_mask
            )
            
            # 投影到 token 空間
            logits = self.output_projection(encoder_output)  # [B, T, vocab_size]
            predicted_tokens = torch.argmax(logits, dim=-1)  # [B, T]
            
            # 移除 padding：找到原始輸入的實際長度
            # 注意：src_tokens 可能已經被填充了，需要找到原始長度
            original_length = src_tokens.size(1)
            if original_length < predicted_tokens.size(1):
                predicted_tokens = predicted_tokens[:, :original_length]
            
            return predicted_tokens
    
    def forward(self, noisy_audio, clean_audio=None, return_logits=False):
        """完整的前向傳播：Audio → Tokens → Transformer → Tokens → Audio
        
        Args:
            noisy_audio: 輸入的噪聲音頻 [B, 1, T]
            clean_audio: 乾淨音頻（訓練/驗證時提供）[B, 1, T]
            return_logits: 強制返回 logits 格式（用於驗證），即使在 eval 模式也返回 logits
                         - True: 返回 {'logits', 'target_tokens', 'noisy_tokens', 'clean_tokens'}
                         - False (推理): 返回 {'denoised_audio', 'denoised_tokens', 'noisy_tokens'}
        
        Returns:
            dict: 根據模式返回不同的內容
                訓練/驗證模式 (training=True 或 return_logits=True):
                    - logits: [B, L, 4096] Transformer 輸出的 token 機率分佈
                    - target_tokens: [B, L] 目標 token 序列
                    - noisy_tokens: [B, L-1] 噪聲 token 序列（已調整長度）
                    - clean_tokens: [B, L_original] 原始乾淨 token 序列
                推理模式 (training=False 且 return_logits=False):
                    - denoised_audio: [B, 1, T] 降噪後的音頻
                    - denoised_tokens: [B, L] 預測的 token 序列
                    - noisy_tokens: [B, L] 原始噪聲 token 序列
        """
        
        # Step 1: 將音頻轉換為 tokens (使用凍結的 WavTokenizer Encoder)
        noisy_tokens = self.encode_audio_to_tokens(noisy_audio)
        
        # 修改條件：訓練模式 OR 需要返回 logits（用於驗證）
        # 這樣在 model.eval() + return_logits=True 時也能計算 loss
        if (self.training or return_logits) and clean_audio is not None:
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
            # 傳遞 return_logits 參數確保在 eval 模式也返回 logits
            logits = self.forward_transformer(input_tokens, decoder_input, return_logits=return_logits)
            
            # 動態調整所有序列到實際的logits長度，而不是強制到max_length
            actual_seq_len = logits.shape[1]
            
            # 確保 target_tokens shape 與 logits 一致
            if target_tokens.shape[1] < actual_seq_len:
                pad_size = actual_seq_len - target_tokens.shape[1]
                pad = torch.full((target_tokens.shape[0], pad_size), self.pad_token, device=target_tokens.device, dtype=target_tokens.dtype)
                target_tokens = torch.cat([target_tokens, pad], dim=1)
            elif target_tokens.shape[1] > actual_seq_len:
                target_tokens = target_tokens[:, :actual_seq_len]
            
            # 同樣調整 input_tokens 的長度
            if input_tokens.shape[1] < actual_seq_len:
                pad_size = actual_seq_len - input_tokens.shape[1]
                pad = torch.full((input_tokens.shape[0], pad_size), self.pad_token, device=input_tokens.device, dtype=input_tokens.dtype)
                input_tokens = torch.cat([input_tokens, pad], dim=1)
            elif input_tokens.shape[1] > actual_seq_len:
                input_tokens = input_tokens[:, :actual_seq_len]
                
            # 調整 noisy_tokens 長度，確保與處理後的序列一致
            # 由於input_tokens = noisy_tokens + EOS，所以noisy_tokens長度應該是actual_seq_len-1
            target_noisy_len = max(1, actual_seq_len - 1)  # 至少保留1個token
            if noisy_tokens.shape[1] < target_noisy_len:
                pad_size = target_noisy_len - noisy_tokens.shape[1]
                pad = torch.full((noisy_tokens.shape[0], pad_size), self.pad_token, device=noisy_tokens.device, dtype=noisy_tokens.dtype)
                noisy_tokens_adjusted = torch.cat([noisy_tokens, pad], dim=1)
            elif noisy_tokens.shape[1] > target_noisy_len:
                noisy_tokens_adjusted = noisy_tokens[:, :target_noisy_len]
            else:
                noisy_tokens_adjusted = noisy_tokens
                
            # Debug: 檢查返回前的形狀
            logger.info(f"Forward return shapes - logits: {logits.shape}, target_tokens: {target_tokens.shape}")
            
            return {
                'logits': logits,
                'target_tokens': target_tokens,
                'noisy_tokens': noisy_tokens_adjusted,  # 使用調整後的版本（長度少1）
                'input_tokens': input_tokens,  # 完整輸入序列（與 target_tokens 長度一致）
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
    """使用原始 CrossEntropy 訓練一個 epoch
    
    優化（2025/10/21）：
        - 使用 CrossEntropyLoss(ignore_index=pad_token) 自動處理 padding
        - 簡化損失計算邏輯，與 validate_epoch 完全一致
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # batch 是 tuple 格式: (noisy_audio, clean_audio, content_id)
        noisy_audio = batch[0].to(device)
        clean_audio = batch[1].to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        output = model(noisy_audio, clean_audio)
        
        logits = output['logits']  # [B, L, vocab_size]
        target_tokens = output['target_tokens']  # [B, L]
        
        # 計算損失 - 使用 reshape 確保 tensor 連續性
        logits_flat = logits.reshape(-1, logits.size(-1))  # [B*L, vocab_size]
        target_flat = target_tokens.reshape(-1)             # [B*L]
        
        # 確保數據類型正確
        # logits 必須是 float，target 必須是 long
        logits_flat = logits_flat.float()
        target_flat = target_flat.long()
        
        # 使用 CrossEntropyLoss 的 ignore_index 參數自動處理 PAD token
        # criterion 已經設置為 nn.CrossEntropyLoss(ignore_index=model.pad_token)
        loss = criterion(logits_flat, target_flat)
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 統計（用於顯示）
        total_loss += loss.item()
        
        # 計算準確率（只在非 PAD token 上）
        mask = target_flat != model.pad_token
        if mask.sum() > 0:
            predictions = torch.argmax(logits_flat[mask], dim=-1)
            total_correct += (predictions == target_flat[mask]).sum().item()
            total_tokens += mask.sum().item()
        
        # 更新進度條
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}', 
            'Acc': f'{accuracy:.1f}%',
            'Tokens': total_tokens
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    
    return avg_loss

def train_epoch_with_token_loss(model, dataloader, optimizer, device, epoch, 
                               loss_weights={'l2': 0.3, 'consistency': 0.4, 'manifold': 0.1, 
                                           'normalization': 0.1, 'coherence': 0.1}):
    """使用 Token Loss 系統訓練一個 epoch（ttt2.py 損失邏輯移植到離散空間）
    
    新增詳細 logging：記錄 CE Loss、Token Accuracy 等關鍵指標
    """
    model.train()
    total_losses = {'total': 0.0}
    loss_counts = 0
    
    # 新增：統計 token 準確率
    total_correct_tokens = 0
    total_tokens_count = 0
    
    # 獲取嵌入層用於 Token Loss 計算
    # 使用全局的 EmbeddingWrapper 來包裝模型的 get_token_embeddings 方法
    embedding_layer = EmbeddingWrapper(model.get_token_embeddings) if hasattr(model, 'get_token_embeddings') else None
    
    if embedding_layer is None:
        logging.warning("無法找到嵌入層，將使用簡化的 token loss")
    else:
        logging.info(f"✅ 找到嵌入層：使用 get_token_embeddings 方法（包含 codebook + special tokens + projection）")
    
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
        input_tokens = output['input_tokens']  # [batch_size, seq_len] - 與 target 長度一致
        
        # 獲取預測 tokens
        predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        
        # 新增：計算 token 準確率
        correct = (predicted_tokens == target_tokens).sum().item()
        total = target_tokens.numel()
        total_correct_tokens += correct
        total_tokens_count += total
        
        # 使用 Token Loss 系統計算損失（改良版：4 個損失組件）
        # 注意：不再 clamp tokens，因為 input_tokens 和 target_tokens 已包含 special tokens (4096-4098)
        try:
            total_loss, loss_dict = compute_combined_token_loss(
                predicted_logits=logits,
                target_tokens=target_tokens,
                input_tokens=input_tokens,  # 使用完整的 input_tokens，長度與 target 一致
                embedding_layer=embedding_layer,
                weights=loss_weights
            )
        except Exception as e:
            logging.warning(f"Token loss 計算失敗，回退到交叉熵: {e}")
            # 回退到簡單交叉熵 - 使用reshape確保tensor連續性
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_tokens.reshape(-1)
            
            # 確保target_flat在有效範圍內 - 修復CUDA斷言錯誤
            target_flat = torch.clamp(target_flat, 0, model.codebook_size - 1)
            
            mask = target_flat < model.codebook_size
            if mask.sum() > 0:
                total_loss = F.cross_entropy(logits_flat[mask], target_flat[mask])
            else:
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict = {'total_loss': total_loss.item(), 'consistency_loss': total_loss.item()}
        
        # 反向傳播
        total_loss.backward()
        
        # 改進的梯度裁剪和穩定化策略
        grad_norm = apply_advanced_gradient_clipping(model, max_norm=0.5, adaptive=True)
        
        optimizer.step()
        
        # 梯度統計記錄
        loss_dict['grad_norm'] = grad_norm
        
        # 累積統計
        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value
        
        loss_counts += 1
        
        # 新增：每 100 個 batch 記錄一次詳細統計
        if (batch_idx + 1) % 100 == 0:
            avg_acc = (total_correct_tokens / total_tokens_count * 100) if total_tokens_count > 0 else 0.0
            logging.info(f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}: "
                        f"Token Accuracy={avg_acc:.2f}%, "
                        f"CE Loss={loss_dict.get('ce_loss', loss_dict.get('consistency_loss', 0.0)):.4f}")
        
        # 更新進度條
        avg_total_loss = total_losses.get('total_loss', total_losses['total']) / loss_counts
        current_acc = (correct / total * 100) if total > 0 else 0.0
        progress_info = {
            'Total': f'{avg_total_loss:.4f}',
            'Acc': f'{current_acc:.1f}%'
        }
        
        # 添加主要 loss 組件到進度條
        if 'consistency_loss' in total_losses:
            progress_info['CE'] = f'{total_losses["consistency_loss"]/loss_counts:.4f}'
        if 'l2_loss' in total_losses:
            progress_info['L2'] = f'{total_losses["l2_loss"]/loss_counts:.4f}'
        
        progress_bar.set_postfix(progress_info)
    
    # 計算平均損失
    avg_losses = {key: value / loss_counts for key, value in total_losses.items()}
    
    # 新增：添加整體 token 準確率到返回值
    avg_losses['token_accuracy'] = (total_correct_tokens / total_tokens_count * 100) if total_tokens_count > 0 else 0.0
    
    # 新增：記錄 epoch 總結
    logging.info(f"Epoch {epoch} Summary: Token Accuracy={avg_losses['token_accuracy']:.2f}%, "
                f"CE Loss={avg_losses.get('ce_loss', avg_losses.get('consistency_loss', 0.0)):.4f}, "
                f"Total Loss={avg_losses.get('total_loss', avg_losses['total']):.4f}")
    
    return avg_losses

def validate_epoch(model, dataloader, criterion, device):
    """驗證一個 epoch
    
    優化設計（2025/10/21 改進）：
        - 使用 model.eval() 設置評估模式（開頭設置一次即可）
        - 通過 return_logits=True 強制返回 logits，即使在 eval 模式
        - 使用 CrossEntropyLoss(ignore_index=pad_token) 自動處理 padding
        - 簡化損失計算邏輯，與 train_epoch 完全一致
        - 保持梯度關閉 (torch.no_grad())，節省內存
    
    Args:
        model: WavTokenizerTransformerDenoiser 模型
        dataloader: 驗證數據加載器
        criterion: 損失函數，應該是 nn.CrossEntropyLoss(ignore_index=model.pad_token)
        device: 計算設備
    
    Returns:
        tuple: (avg_loss, accuracy)
            - avg_loss: 平均驗證損失，若無有效batch則返回 NaN
            - accuracy: Token 預測準確率 (0.0-1.0)
    """
    model.eval()  # 設置為評估模式
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    valid_batches = 0
    
    # 限制驗證批次數量以節省內存
    max_val_batches = 50
    
    with torch.no_grad():  # 確保不計算梯度
        for batch in tqdm(dataloader, desc="Validation"):
            if valid_batches >= max_val_batches:
                break
            
            try:
                # batch 是 tuple 格式: (noisy_audio, clean_audio, content_id)
                noisy_audio, clean_audio, _ = batch
                noisy_audio = noisy_audio.to(device)
                clean_audio = clean_audio.to(device)
                
                # >>>>> 核心改動 <<<<<
                # 直接調用 forward，並傳遞 return_logits=True
                # 模型內部會處理所有 tokenization 和序列準備工作
                output = model(noisy_audio, clean_audio, return_logits=True)
                
                logits = output['logits']              # [B, L, 4096]
                target_tokens = output['target_tokens'] # [B, L]
                
                # Debug: 檢查形狀（使用 INFO 級別確保輸出）
                logger.info(f"Validation shapes - logits: {logits.shape}, target: {target_tokens.shape}")
                
                # --- 計算損失和準確率 (與 train_epoch 完全一致) ---
                # 使用與 train 完全相同的 flatten 方式
                logits_flat = logits.view(-1, logits.size(-1))  # [B*L, vocab_size]
                target_flat = target_tokens.view(-1)             # [B*L]
                
                logger.info(f"After flatten - logits_flat: {logits_flat.shape}, target_flat: {target_flat.shape}")
                
                # 確保數據類型正確
                # logits 必須是 float，target 必須是 long
                logits_flat = logits_flat.float()
                target_flat = target_flat.long()
                
                # 使用 CrossEntropyLoss 的 ignore_index 參數自動處理 PAD token
                # criterion 應該已經設置為 nn.CrossEntropyLoss(ignore_index=model.pad_token)
                loss = criterion(logits_flat, target_flat)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    # 計算準確率 (只在非 PAD token 上計算)
                    mask = target_flat != model.pad_token
                    if mask.sum() > 0:
                        predictions = torch.argmax(logits_flat[mask], dim=-1)
                        total_correct += (predictions == target_flat[mask]).sum().item()
                        total_tokens += mask.sum().item()
                
            except Exception as e:
                logging.error(f"驗證批次出錯，跳過: {e}")
                import traceback
                traceback.print_exc()  # 打印詳細的錯誤堆棧
                continue
    
    # 計算平均損失和準確率
    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
    else:
        logging.error("驗證過程中沒有有效的batch")
        avg_loss = float('nan')  # 使用 NaN 表示驗證失敗
    
    accuracy = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    
    # 記錄詳細的驗證結果
    if valid_batches > 0:
        logging.info(f"Validation: Loss={avg_loss:.4f}, Token Accuracy={accuracy*100:.2f}%, Valid Batches={valid_batches}")
    
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


def save_checkpoint(model, optimizer, epoch, loss, save_path, config=None, train_losses=None, val_losses=None, val_accuracies=None):
    """保存模型檢查點"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }
    
    # 添加訓練歷史到檢查點
    if train_losses is not None:
        checkpoint['train_losses'] = train_losses
    if val_losses is not None:
        checkpoint['val_losses'] = val_losses
    if val_accuracies is not None:
        checkpoint['val_accuracies'] = val_accuracies
        
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
    parser.add_argument('--disable_scheduler', action='store_true', help='禁用學習率調度器（使用固定 LR）')
    parser.add_argument('--token_loss_weight', type=float, default=1.0, help='Token Loss權重')
    
    # 損失函數選擇（重構版 - 更清晰的權重設計）
    parser.add_argument('--use_token_loss', action='store_true', 
                        help='使用混合 Token Loss 系統（CE + L2_Embed + Coherence + Manifold）')
    parser.add_argument('--ce_weight', type=float, default=10.0, 
                        help='交叉熵損失權重（主要監督信號，確保預測準確）[修改：1.0→10.0 修復 token 準確率問題]')
    parser.add_argument('--l2_embed_weight', type=float, default=0.5, 
                        help='L2 Embedding 損失權重（聲學相似性，錯也要錯得像）')
    parser.add_argument('--coherence_weight', type=float, default=0.2, 
                        help='連貫性損失權重（時間平滑，解決頻譜破碎）')
    parser.add_argument('--manifold_weight', type=float, default=0.1, 
                        help='Manifold 正則化權重（防止偏離輸入太遠）')

    # 數據參數
    parser.add_argument('--output_dir', type=str, default='results/wavtokenizer_transformer_denoising',
                        help='輸出目錄')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大處理樣本數 (None 表示使用全部數據)')
    parser.add_argument('--max_sentences_per_speaker', type=int, default=None,
                        help='每位語者最大句子數')
    parser.add_argument('--val_speakers', nargs='+', default=['girl9', 'boy7'],
                        help='驗證集語者')
    parser.add_argument('--train_speakers', nargs='+', default=None,
                        help='訓練集語者 (如果指定，只使用這些語者進行訓練；如果不指定，使用除驗證集外的所有語者)')
    
    # 檢查點恢復參數
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='從指定檢查點恢復訓練的路徑')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='開始訓練的epoch數 (用於檢查點恢復)')
    
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
    
    # 準備允許的語者列表（訓練語者 + 驗證語者）
    allowed_speakers = set()
    if args.train_speakers:
        allowed_speakers.update(args.train_speakers)
    if args.val_speakers:
        allowed_speakers.update(args.val_speakers)
    
    # 如果沒有指定訓練語者，預設允許所有語者
    if not args.train_speakers:
        allowed_speakers = None  # None 表示允許所有語者
    else:
        allowed_speakers = list(allowed_speakers)
        logging.info(f"只載入以下語者的數據: {allowed_speakers}")
    
    # 創建音頻數據集，只載入需要的語者
    audio_dataset = AudioDataset(input_dirs, target_dir, 
                                max_sentences_per_speaker=args.max_sentences_per_speaker,
                                allowed_speakers=allowed_speakers)
    
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
    
    # 損失函數：使用 ignore_index 自動處理 PAD token
    # 這樣就不需要在損失計算時手動創建 mask 了
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token)
    logging.info(f"✅ CrossEntropyLoss 已設置 ignore_index={model.pad_token} (PAD token)")
    
    # 學習率調度器
    if args.disable_scheduler:
        # 使用固定 LR（不進行調度）
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=args.num_epochs * len(train_loader)
        )
        logging.info(f"🔧 使用固定 Learning Rate: {args.learning_rate}")
    else:
        # 使用 OneCycleLR
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=args.num_epochs,
            pct_start=0.1
        )
        logging.info(f"🔧 使用 OneCycleLR scheduler (max_lr={args.learning_rate})")
    
    # 準備損失權重字典（重構版 - 更清晰的命名）
    loss_weights = {
        'ce': args.ce_weight,              # 主要監督信號 (1.0)
        'l2_embed': args.l2_embed_weight,  # 聲學相似性 (0.5)
        'coherence': args.coherence_weight, # 時間平滑 (0.2)
        'manifold': args.manifold_weight   # 正則化 (0.1)
    }
    
    # 訓練循環
    logging.info("開始訓練...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    start_epoch = 1
    
    # 檢查點恢復邏輯
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logging.info(f"從檢查點恢復訓練: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
        # 載入模型狀態
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("已載入模型狀態")
        
        # 載入優化器狀態
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("已載入優化器狀態")
        
        # 設置起始epoch
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        
        # 如果檢查點包含訓練歷史，也恢復它們
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
        if 'val_accuracies' in checkpoint:
            val_accuracies = checkpoint['val_accuracies']
            
        logging.info(f"恢復訓練從epoch {start_epoch}開始，當前最佳驗證loss: {best_val_loss:.4f}")
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        try:
            # 選擇訓練函數
            if args.use_token_loss:
                # 使用 Token Loss 系統（ttt2.py 邏輯移植到離散空間）
                train_loss_dict = train_epoch_with_token_loss(
                    model, train_loader, optimizer, device, epoch, loss_weights
                )
                train_loss = train_loss_dict.get('total_loss', 0.0)
                
                # 記錄詳細的損失信息（新增：包含 Token Accuracy 和 CE Loss）
                ce_loss = train_loss_dict.get('ce_loss', train_loss_dict.get('consistency_loss', 0.0))
                token_acc = train_loss_dict.get('token_accuracy', 0.0)
                
                loss_info = []
                loss_info.append(f"Total={train_loss:.4f}")
                loss_info.append(f"CE={ce_loss:.4f}")
                loss_info.append(f"TokenAcc={token_acc:.2f}%")
                
                # 添加其他損失組件
                for k, v in train_loss_dict.items():
                    if k not in ['total_loss', 'total', 'ce_loss', 'consistency_loss', 'token_accuracy']:
                        loss_info.append(f"{k}={v:.4f}")
                
                logging.info(f"Epoch {epoch} Train Loss: {' | '.join(loss_info)}")
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

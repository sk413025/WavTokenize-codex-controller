#!/usr/bin/env python3
"""
TTT2 Token-Based Feature Enhancement System
基於 WavTokenizer 的 Token 空間特徵增強系統

實驗目的：
1. WavTokenizer 的 encoder/decoder (凍結) 作為基準
2. Input 為不同材質（噪音）下不同語者的說話音檔
3. Target 為不同語者的乾淨音檔
4. 通過特徵增強層學習從 noisy tokens → clean tokens 的映射
5. 目標：通用於不同材質、語者，還原出該語者乾淨音檔的模型

架構：
    Noisy Audio → WavTokenizer Encoder (凍結) → Noisy Tokens
                                                      ↓
    Target Audio → WavTokenizer Encoder (凍結) → Target Tokens (參考)
                                                      ↓
                                            Feature Enhancement Layer
                                                   (可訓練)
                                                      ↓
                                              Enhanced Tokens
                                                      ↓
                                    WavTokenizer Decoder (凍結) → Clean Audio

關鍵修復：
- 正確的 token → feature → audio 重建流程
- 修復 codes_to_features 的維度問題
- 確保 decoder 輸入格式正確
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchaudio
from datetime import datetime

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder.pretrained import WavTokenizer
from ttdata import AudioDataset
from encoder.utils import convert_audio

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """設定隨機種子以確保實驗可重現"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"已設定隨機種子為: {seed}")


class TokenFeatureEnhancer(nn.Module):
    """
    Token 特徵增強層
    
    功能：在 token embedding 空間進行特徵增強
    - 輸入：noisy token embeddings
    - 輸出：enhanced token embeddings
    - 參考：target token embeddings (用於計算損失)
    """
    def __init__(self, 
                 embed_dim=512,
                 num_layers=4,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='gelu'):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Transformer Encoder 層用於特徵增強
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # 可選的後處理層
        self.post_process = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 殘差連接的權重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        logger.info(f"TokenFeatureEnhancer 初始化完成: embed_dim={embed_dim}, num_layers={num_layers}")
    
    def forward(self, noisy_features, mask=None):
        """
        Args:
            noisy_features: [batch_size, seq_len, embed_dim]
            mask: Optional padding mask
            
        Returns:
            enhanced_features: [batch_size, seq_len, embed_dim]
        """
        # Transformer 增強
        enhanced = self.transformer_encoder(noisy_features, src_key_padding_mask=mask)
        
        # 後處理
        enhanced = self.post_process(enhanced)
        
        # 殘差連接
        enhanced_features = noisy_features + self.residual_weight * enhanced
        
        return enhanced_features


class TTT2TokenModel(nn.Module):
    """
    完整的 Token-based 特徵增強模型
    
    流程：
    1. Noisy Audio → WavTokenizer Encoder (凍結) → Noisy Tokens
    2. Noisy Tokens → Embedding → Noisy Features
    3. Noisy Features → Feature Enhancer (可訓練) → Enhanced Features
    4. Enhanced Features → Quantize to Tokens → Enhanced Tokens
    5. Enhanced Tokens → WavTokenizer Decoder (凍結) → Enhanced Audio
    """
    def __init__(self,
                 config_path,
                 model_path,
                 embed_dim=512,
                 enhancer_layers=4,
                 enhancer_heads=8,
                 enhancer_ff_dim=2048,
                 dropout=0.1,
                 device='cuda'):
        super().__init__()
        
        self.device = device
        self.embed_dim = embed_dim
        
        # 1. 載入預訓練的 WavTokenizer (凍結)
        logger.info("載入預訓練的 WavTokenizer...")
        self.wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.wavtokenizer = self.wavtokenizer.to(device)  # 移到指定設備
        self.wavtokenizer.eval()
        
        # 凍結 WavTokenizer
        for param in self.wavtokenizer.parameters():
            param.requires_grad = False
        
        self.codebook_size = 4096  # WavTokenizer codebook size
        
        # 2. 直接使用 WavTokenizer 的 Codebook 作為 Token Embedding
        # 不創建新的 nn.Embedding，而是提取預訓練的 codebook weights
        logger.info("提取 WavTokenizer 的 codebook 作為 token embedding...")
        self.codebook_weights = self._extract_codebook_weights()
        
        # 如果 codebook 維度與 embed_dim 不同，需要投影層
        codebook_dim = self.codebook_weights.shape[1]
        if codebook_dim != embed_dim:
            self.codebook_projection = nn.Linear(codebook_dim, embed_dim)
            logger.info(f"Codebook 維度投影: {codebook_dim} → {embed_dim}")
        else:
            self.codebook_projection = nn.Identity()
            logger.info(f"Codebook 維度匹配: {codebook_dim}")
        
        # 3. Positional Encoding
        self.pos_encoding = self._create_positional_encoding(max_len=1000, d_model=embed_dim)
        
        # 4. Feature Enhancer (可訓練的核心模組)
        self.feature_enhancer = TokenFeatureEnhancer(
            embed_dim=embed_dim,
            num_layers=enhancer_layers,
            nhead=enhancer_heads,
            dim_feedforward=enhancer_ff_dim,
            dropout=dropout
        )
        
        # 5. Feature to Token (特徵映射回 codebook 空間用於量化)
        # 將增強的特徵投影到 codebook 的特徵維度
        # 然後通過最近鄰搜索找到對應的 token
        codebook_dim = self.codebook_weights.shape[1]
        self.feature_to_codebook = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, codebook_dim)  # 投影到 codebook 的特徵空間
        )
        
        logger.info(f"Feature to Codebook projection: {embed_dim} → {codebook_dim}")
        
        logger.info(f"TTT2TokenModel 初始化完成")
        logger.info(f"- Codebook size: {self.codebook_size}")
        logger.info(f"- Embed dim: {embed_dim}")
        logger.info(f"- Enhancer layers: {enhancer_layers}")
        
        # 統計參數
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"- 總參數量: {total_params:,}")
        logger.info(f"- 可訓練參數量: {trainable_params:,}")
    
    def _extract_codebook_weights(self):
        """
        從 WavTokenizer 提取預訓練的 codebook weights
        
        這是關鍵：直接使用 WavTokenizer 學到的 token 表示，
        而不是創建一個隨機初始化的 Embedding 層
        
        Returns:
            codebook_weights: [total_codebook_size, codebook_dim]
        """
        try:
            # 從 WavTokenizer 的 quantizer 提取 codebook
            # 參考: decoder/pretrained.py line 239
            vq_layers = self.wavtokenizer.feature_extractor.encodec.quantizer.vq.layers
            
            # 將所有層的 codebook 拼接起來
            codebook_weights = torch.cat([vq.codebook for vq in vq_layers], dim=0)
            
            logger.info(f"成功提取 codebook weights: shape={codebook_weights.shape}")
            logger.info(f"- 來自 {len(vq_layers)} 個 VQ 層")
            logger.info(f"- 總共 {codebook_weights.shape[0]} 個 codes")
            logger.info(f"- 每個 code 維度: {codebook_weights.shape[1]}")
            
            # 凍結 codebook weights (保持預訓練的語義)
            codebook_weights = codebook_weights.detach()
            
            return codebook_weights
            
        except Exception as e:
            logger.error(f"提取 codebook weights 失敗: {e}")
            raise
    
    def _create_positional_encoding(self, max_len, d_model):
        """創建位置編碼"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def quantize_features_to_tokens(self, features):
        """
        將連續特徵量化為 discrete tokens
        通過在 codebook 中尋找最近鄰
        
        Args:
            features: [batch_size, seq_len, codebook_dim]
            
        Returns:
            tokens: [batch_size, seq_len] - token indices
        """
        batch_size, seq_len, dim = features.shape
        
        # 將特徵展平以便批次處理
        flat_features = features.reshape(-1, dim)  # [B*L, dim]
        
        # 計算與所有 codebook entries 的歐氏距離
        # codebook_weights: [codebook_size, dim]
        # flat_features: [B*L, dim]
        distances = torch.cdist(flat_features, self.codebook_weights)  # [B*L, codebook_size]
        
        # 找最近的 codebook entry
        tokens = torch.argmin(distances, dim=-1)  # [B*L]
        
        # 重塑回原始形狀
        tokens = tokens.reshape(batch_size, seq_len)  # [B, L]
        
        return tokens
    
    def encode_audio_to_tokens(self, audio):
        """
        使用 WavTokenizer Encoder 將音頻轉換為 tokens
        
        Args:
            audio: [batch_size, 1, time] or [batch_size, time]
            
        Returns:
            tokens: [batch_size, seq_len] - discrete token indices
            features: [batch_size, channels, seq_len] - continuous features
        """
        with torch.no_grad():
            # 確保音頻維度正確
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # [batch_size, 1, time]
            
            # 使用 WavTokenizer 編碼
            bandwidth_id = torch.tensor([0], device=audio.device)
            features, discrete_code = self.wavtokenizer.encode_infer(audio, bandwidth_id=bandwidth_id)
            
            # discrete_code shape: [n_q, batch_size, seq_len]
            # 我們只使用第一層的 codes (最重要的層)
            tokens = discrete_code[0]  # [batch_size, seq_len]
            
            # 重要：clone tokens 以便用於 autograd
            # inference tensors 不能直接用於反向傳播
            tokens = tokens.clone().detach()
            features = features.clone().detach()
            
            return tokens, features
    
    def decode_tokens_to_audio(self, tokens):
        """
        使用 WavTokenizer Decoder 將 tokens 重建為音頻
        
        **修復關鍵**: 正確處理 token → feature → audio 的轉換
        
        Args:
            tokens: [batch_size, seq_len] - discrete token indices
            
        Returns:
            audio: [batch_size, 1, time]
        """
        with torch.no_grad():
            # 準備 discrete codes 的正確格式
            # WavTokenizer 期望 codes 為 [n_q, batch_size, seq_len]
            # 我們只有單層 codebook，所以 n_q = 1
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(0)  # [1, batch_size, seq_len]
            
            # 確保 tokens 在有效範圍內
            tokens = torch.clamp(tokens, 0, self.codebook_size - 1)
            
            # 使用 codes_to_features 轉換為連續特徵
            features = self.wavtokenizer.codes_to_features(tokens)
            
            # 使用 decoder 重建音頻
            bandwidth_id = torch.tensor([0], device=tokens.device)
            audio = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
            
            # 確保音頻維度正確 [batch_size, 1, time]
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            elif audio.dim() == 3 and audio.size(1) != 1:
                # 如果是 [batch_size, channels, time] 且 channels != 1
                audio = audio.mean(dim=1, keepdim=True)  # 轉為單聲道
            
            return audio
    
    def forward(self, noisy_audio, target_audio=None, return_intermediate=False):
        """
        完整的前向傳播
        
        Args:
            noisy_audio: [batch_size, 1, time] 噪音音頻
            target_audio: [batch_size, 1, time] 目標乾淨音頻 (訓練時需要)
            return_intermediate: 是否返回中間結果
            
        Returns:
            訓練模式: dict with loss components
            推理模式: dict with enhanced audio
        """
        # Step 1: 編碼音頻為 tokens
        noisy_tokens, noisy_wavtok_features = self.encode_audio_to_tokens(noisy_audio)
        
        if target_audio is not None:
            target_tokens, target_wavtok_features = self.encode_audio_to_tokens(target_audio)
        
        # Step 2: Token Embedding (使用預訓練的 codebook weights)
        batch_size, seq_len = noisy_tokens.shape
        
        # 使用 WavTokenizer 的 codebook 作為 embedding
        # 這保留了預訓練模型學到的語義結構
        noisy_emb = F.embedding(noisy_tokens, self.codebook_weights)  # [batch_size, seq_len, codebook_dim]
        
        # 如果需要，投影到增強空間的維度
        noisy_emb = self.codebook_projection(noisy_emb)  # [batch_size, seq_len, embed_dim]
        
        # 添加位置編碼
        if seq_len <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :seq_len, :].to(noisy_emb.device)
        else:
            # 如果序列太長，擴展位置編碼
            pos_enc = self._create_positional_encoding(seq_len, self.embed_dim).to(noisy_emb.device)
        
        noisy_features = noisy_emb + pos_enc
        
        # Step 3: Feature Enhancement (可訓練)
        enhanced_features = self.feature_enhancer(noisy_features)
        
        # Step 4: 將增強的特徵映射回 codebook 空間並量化為 tokens
        # 4.1 投影到 codebook 的特徵維度
        enhanced_codebook_features = self.feature_to_codebook(enhanced_features)  # [B, L, codebook_dim]
        
        # 4.2 通過最近鄰搜索量化為 discrete tokens
        # 計算與所有 codebook entries 的距離
        enhanced_tokens = self.quantize_features_to_tokens(enhanced_codebook_features)  # [B, L]
        
        # Step 5: 使用 Decoder 重建音頻
        enhanced_audio = self.decode_tokens_to_audio(enhanced_tokens)
        
        # 準備返回結果
        result = {
            'enhanced_audio': enhanced_audio,
            'enhanced_tokens': enhanced_tokens,
            'noisy_tokens': noisy_tokens,
            'enhanced_codebook_features': enhanced_codebook_features
        }
        
        if target_audio is not None:
            # 訓練模式：計算目標 (同樣使用預訓練的 codebook)
            target_emb = F.embedding(target_tokens, self.codebook_weights)
            target_emb = self.codebook_projection(target_emb)
            target_features = target_emb + pos_enc
            
            result.update({
                'target_tokens': target_tokens,
                'target_features': target_features,
                'enhanced_features': enhanced_features,
                'noisy_features': noisy_features,
                'target_audio_reconstructed': self.decode_tokens_to_audio(target_tokens)
            })
        
        if return_intermediate:
            result['intermediate'] = {
                'noisy_wavtok_features': noisy_wavtok_features,
                'target_wavtok_features': target_wavtok_features if target_audio is not None else None
            }
        
        return result


def compute_token_enhancement_loss(model_output, loss_weights=None):
    """
    計算 Token Enhancement 損失
    
    損失組件:
    1. Token Classification Loss (CrossEntropy): 預測正確的 target tokens
    2. Feature L2 Loss: enhanced features 接近 target features
    3. Token Consistency Loss: 相同內容的 tokens 應該相似
    4. Audio Reconstruction Loss: 重建的音頻接近目標音頻
    """
    if loss_weights is None:
        loss_weights = {
            'token_ce': 0.4,        # Token分類損失
            'feature_l2': 0.3,      # 特徵L2損失
            'audio_l1': 0.2,        # 音頻L1損失
            'token_smooth': 0.1     # Token平滑損失
        }
    
    losses = {}
    
    # 1. Token Classification Loss (CrossEntropy)
    token_logits = model_output['token_logits']  # [batch, seq_len, vocab]
    target_tokens = model_output['target_tokens']  # [batch, seq_len]
    
    token_logits_flat = token_logits.reshape(-1, token_logits.size(-1))
    target_tokens_flat = target_tokens.reshape(-1)
    
    losses['token_ce'] = F.cross_entropy(token_logits_flat, target_tokens_flat, ignore_index=-1)
    
    # 2. Feature L2 Loss
    enhanced_features = model_output['enhanced_features']
    target_features = model_output['target_features']
    losses['feature_l2'] = F.mse_loss(enhanced_features, target_features)
    
    # 3. Audio Reconstruction L1 Loss
    enhanced_audio = model_output['enhanced_audio']
    target_audio_reconstructed = model_output['target_audio_reconstructed']
    
    # 確保音頻長度一致
    min_len = min(enhanced_audio.size(-1), target_audio_reconstructed.size(-1))
    enhanced_audio_trimmed = enhanced_audio[..., :min_len]
    target_audio_trimmed = target_audio_reconstructed[..., :min_len]
    
    losses['audio_l1'] = F.l1_loss(enhanced_audio_trimmed, target_audio_trimmed)
    
    # 4. Token Smoothness Loss (防止 token 序列過於突變)
    enhanced_tokens = model_output['enhanced_tokens'].float()
    token_diff = enhanced_tokens[:, 1:] - enhanced_tokens[:, :-1]
    losses['token_smooth'] = torch.mean(torch.abs(token_diff))
    
    # 計算總損失
    total_loss = sum(loss_weights[k] * losses[k] for k in losses.keys())
    losses['total'] = total_loss
    
    return losses, total_loss


def save_audio_sample(audio, save_path, sample_rate=24000):
    """保存音頻樣本"""
    try:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu()
            if audio.dim() == 3:
                audio = audio.squeeze(0)
            if audio.dim() == 2:
                audio = audio.squeeze(0)
        
        torchaudio.save(save_path, audio.unsqueeze(0), sample_rate=sample_rate)
    except Exception as e:
        logger.error(f"保存音頻失敗: {e}")


def save_spectrogram(audio, save_path, title="Spectrogram"):
    """保存頻譜圖"""
    try:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().squeeze().numpy()
        
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=512,
            n_mels=80
        )
        
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        mel_spec = transform(audio_tensor)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spec_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='dB')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"保存頻譜圖失敗: {e}")


def train_epoch(model, dataloader, optimizer, device, epoch, save_dir, loss_weights=None):
    """訓練一個 epoch"""
    model.train()
    
    total_losses = {'total': 0.0, 'token_ce': 0.0, 'feature_l2': 0.0, 'audio_l1': 0.0, 'token_smooth': 0.0}
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        noisy_audio = batch[0].to(device)
        target_audio = batch[1].to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        model_output = model(noisy_audio, target_audio)
        
        # 計算損失
        losses, total_loss = compute_token_enhancement_loss(model_output, loss_weights)
        
        # 反向傳播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 統計
        for k in total_losses.keys():
            total_losses[k] += losses[k].item()
        num_batches += 1
        
        # 更新進度條
        progress_bar.set_postfix({
            'Loss': f'{losses["total"].item():.4f}',
            'TokenCE': f'{losses["token_ce"].item():.4f}',
            'FeatL2': f'{losses["feature_l2"].item():.4f}',
            'AudioL1': f'{losses["audio_l1"].item():.4f}'
        })
        
        # 保存樣本 (每 50 個 batch)
        if batch_idx % 50 == 0 and batch_idx > 0:
            sample_dir = os.path.join(save_dir, 'audio_samples', f'epoch_{epoch}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存第一個樣本
            save_audio_sample(noisy_audio[0], os.path.join(sample_dir, f'batch_{batch_idx}_noisy.wav'))
            save_audio_sample(target_audio[0], os.path.join(sample_dir, f'batch_{batch_idx}_target.wav'))
            save_audio_sample(model_output['enhanced_audio'][0], os.path.join(sample_dir, f'batch_{batch_idx}_enhanced.wav'))
            
            save_spectrogram(noisy_audio[0], os.path.join(sample_dir, f'batch_{batch_idx}_noisy_spec.png'), 'Noisy')
            save_spectrogram(target_audio[0], os.path.join(sample_dir, f'batch_{batch_idx}_target_spec.png'), 'Target')
            save_spectrogram(model_output['enhanced_audio'][0], os.path.join(sample_dir, f'batch_{batch_idx}_enhanced_spec.png'), 'Enhanced')
    
    # 計算平均損失
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def validate_epoch(model, dataloader, device, epoch, save_dir):
    """驗證一個 epoch"""
    model.eval()
    
    total_losses = {'total': 0.0, 'token_ce': 0.0, 'feature_l2': 0.0, 'audio_l1': 0.0, 'token_smooth': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch}")):
            noisy_audio = batch[0].to(device)
            target_audio = batch[1].to(device)
            
            # 前向傳播
            model_output = model(noisy_audio, target_audio)
            
            # 計算損失
            losses, _ = compute_token_enhancement_loss(model_output)
            
            # 統計
            for k in total_losses.keys():
                total_losses[k] += losses[k].item()
            num_batches += 1
            
            # 保存驗證樣本 (前3個batch)
            if batch_idx < 3:
                sample_dir = os.path.join(save_dir, 'validation_samples', f'epoch_{epoch}')
                os.makedirs(sample_dir, exist_ok=True)
                
                for i in range(min(2, noisy_audio.size(0))):
                    save_audio_sample(noisy_audio[i], os.path.join(sample_dir, f'batch_{batch_idx}_sample_{i}_noisy.wav'))
                    save_audio_sample(target_audio[i], os.path.join(sample_dir, f'batch_{batch_idx}_sample_{i}_target.wav'))
                    save_audio_sample(model_output['enhanced_audio'][i], os.path.join(sample_dir, f'batch_{batch_idx}_sample_{i}_enhanced.wav'))
    
    # 計算平均損失
    avg_losses = {k: v / num_batches if num_batches > 0 else 0.0 for k, v in total_losses.items()}
    
    return avg_losses


def plot_training_history(train_losses, val_losses, save_path):
    """繪製訓練歷史"""
    epochs = range(1, len(train_losses['total']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Loss
    axes[0, 0].plot(epochs, train_losses['total'], 'b-', label='Train')
    axes[0, 0].plot(epochs, val_losses['total'], 'r-', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Token CE Loss
    axes[0, 1].plot(epochs, train_losses['token_ce'], 'b-', label='Train')
    axes[0, 1].plot(epochs, val_losses['token_ce'], 'r-', label='Val')
    axes[0, 1].set_title('Token CrossEntropy Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Feature L2 Loss
    axes[1, 0].plot(epochs, train_losses['feature_l2'], 'b-', label='Train')
    axes[1, 0].plot(epochs, val_losses['feature_l2'], 'r-', label='Val')
    axes[1, 0].set_title('Feature L2 Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Audio L1 Loss
    axes[1, 1].plot(epochs, train_losses['audio_l1'], 'b-', label='Train')
    axes[1, 1].plot(epochs, val_losses['audio_l1'], 'r-', label='Val')
    axes[1, 1].set_title('Audio L1 Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"訓練歷史圖已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='TTT2 Token-Based Feature Enhancement Training')
    
    # 模型參數
    parser.add_argument('--config_path', type=str, 
                       default='./config/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml')
    parser.add_argument('--model_path', type=str,
                       default='./wavtokenizer_small_600_24k_4096.ckpt')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--enhancer_layers', type=int, default=4)
    parser.add_argument('--enhancer_heads', type=int, default=8)
    parser.add_argument('--enhancer_ff_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 數據參數
    parser.add_argument('--train_speakers', nargs='+', 
                       default=['boy1', 'boy3', 'boy4', 'boy5', 'boy6', 
                               'girl2', 'girl3', 'girl4', 'girl6', 'girl7'])
    parser.add_argument('--val_speakers', nargs='+', default=['girl9', 'boy7'])
    parser.add_argument('--max_sentences_per_speaker', type=int, default=None,
                       help='每個語者最多使用的句子數，None表示使用全部')
    
    # 損失權重
    parser.add_argument('--loss_weight_token_ce', type=float, default=0.4)
    parser.add_argument('--loss_weight_feature_l2', type=float, default=0.3)
    parser.add_argument('--loss_weight_audio_l1', type=float, default=0.2)
    parser.add_argument('--loss_weight_token_smooth', type=float, default=0.1)
    
    # 儲存參數
    parser.add_argument('--output_dir', type=str, default='./results/ttt2_token_enhancement')
    parser.add_argument('--save_every', type=int, default=10, help='每N個epoch保存一次模型')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 設定隨機種子
    set_seed(args.seed)
    
    # 創建輸出目錄
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'exp_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    
    logger.info(f"輸出目錄: {save_dir}")
    
    # 保存配置
    import json
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 準備數據
    logger.info("準備數據集...")
    
    # 使用環境變數中的路徑或默認路徑
    input_dirs = ['/home/sbplab/ruizi/DATA/3materials/audio/box/']
    target_dir = '/home/sbplab/ruizi/DATA/3materials/audio/clean/'
    
    # 創建完整數據集
    full_dataset = AudioDataset(
        input_dirs=input_dirs,
        target_dir=target_dir,
        max_sentences_per_speaker=args.max_sentences_per_speaker,
        allowed_speakers=args.train_speakers + args.val_speakers
    )
    
    # 分割訓練集和驗證集
    train_indices = [i for i in range(len(full_dataset)) 
                    if full_dataset.audio_pairs[i][0].split('_')[1] in args.train_speakers]
    val_indices = [i for i in range(len(full_dataset))
                  if full_dataset.audio_pairs[i][0].split('_')[1] in args.val_speakers]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    logger.info(f"訓練集大小: {len(train_dataset)}")
    logger.info(f"驗證集大小: {len(val_dataset)}")
    
    # 創建 DataLoader
    def collate_fn(batch):
        """處理批次數據"""
        noisy_audios = []
        target_audios = []
        content_ids = []
        
        # 找到最短的音頻長度
        min_len = min([item[0].size(-1) for item in batch] + [item[1].size(-1) for item in batch])
        
        for noisy, target, cid in batch:
            # 裁剪到相同長度
            noisy_audios.append(noisy[..., :min_len])
            target_audios.append(target[..., :min_len])
            content_ids.append(cid)
        
        return (
            torch.stack(noisy_audios),
            torch.stack(target_audios),
            content_ids
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 創建模型
    logger.info("創建 TTT2 Token Enhancement 模型...")
    model = TTT2TokenModel(
        config_path=args.config_path,
        model_path=args.model_path,
        embed_dim=args.embed_dim,
        enhancer_layers=args.enhancer_layers,
        enhancer_heads=args.enhancer_heads,
        enhancer_ff_dim=args.enhancer_ff_dim,
        dropout=args.dropout,
        device=args.device
    ).to(args.device)
    
    # 創建優化器和調度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # 損失權重
    loss_weights = {
        'token_ce': args.loss_weight_token_ce,
        'feature_l2': args.loss_weight_feature_l2,
        'audio_l1': args.loss_weight_audio_l1,
        'token_smooth': args.loss_weight_token_smooth
    }
    
    logger.info(f"損失權重: {loss_weights}")
    
    # 訓練歷史
    train_history = {'total': [], 'token_ce': [], 'feature_l2': [], 'audio_l1': [], 'token_smooth': []}
    val_history = {'total': [], 'token_ce': [], 'feature_l2': [], 'audio_l1': [], 'token_smooth': []}
    
    best_val_loss = float('inf')
    
    # 開始訓練
    logger.info("開始訓練...")
    logger.info(f"設備: {args.device}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"總 Epochs: {args.num_epochs}")
    logger.info("=" * 80)
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.num_epochs}")
        logger.info("-" * 80)
        
        # 訓練
        train_losses = train_epoch(model, train_loader, optimizer, args.device, epoch, save_dir, loss_weights)
        
        # 驗證
        val_losses = validate_epoch(model, val_loader, args.device, epoch, save_dir)
        
        # 更新學習率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄歷史
        for k in train_history.keys():
            train_history[k].append(train_losses[k])
            val_history[k].append(val_losses[k])
        
        # 打印結果
        logger.info(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f} | LR: {current_lr:.6f}")
        logger.info(f"  TokenCE: {train_losses['token_ce']:.4f} | {val_losses['token_ce']:.4f}")
        logger.info(f"  FeatL2:  {train_losses['feature_l2']:.4f} | {val_losses['feature_l2']:.4f}")
        logger.info(f"  AudioL1: {train_losses['audio_l1']:.4f} | {val_losses['audio_l1']:.4f}")
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'train_history': train_history,
                'val_history': val_history
            }, os.path.join(save_dir, 'checkpoints', 'best_model.pth'))
            logger.info(f"✅ 保存最佳模型 (Val Loss: {best_val_loss:.4f})")
        
        # 定期保存模型
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total'],
                'train_history': train_history,
                'val_history': val_history
            }, os.path.join(save_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth'))
            logger.info(f"💾 保存檢查點: epoch_{epoch}.pth")
        
        # 繪製訓練歷史
        if epoch % 5 == 0 or epoch == args.num_epochs:
            plot_training_history(
                train_history,
                val_history,
                os.path.join(save_dir, f'training_history_epoch_{epoch}.png')
            )
    
    # 保存最終模型
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_history': train_history,
        'val_history': val_history
    }, os.path.join(save_dir, 'checkpoints', 'final_model.pth'))
    
    logger.info("=" * 80)
    logger.info("訓練完成！")
    logger.info(f"最佳驗證損失: {best_val_loss:.4f}")
    logger.info(f"結果保存在: {save_dir}")


if __name__ == '__main__':
    main()

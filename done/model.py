"""
Token Denoising Transformer 模型
完全重用 WavTokenizer 的 Codebook (凍結，不訓練)
類比機器翻譯: Token IDs in → Frozen Embedding → Transformer → Token IDs out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from tqdm import tqdm
import math


# ============================================================================
#                    Token Denoising Transformer
# ============================================================================

class TokenDenoisingTransformer(nn.Module):
    """
    基於 Frozen Codebook 的 Token Denoising Transformer
    
    架構:
        Noisy Token IDs (B, T)
        → Frozen Codebook Lookup → (B, T, 512)
        → Positional Encoding
        → Transformer Encoder
        → Linear Projection → (B, T, 4096)
        → Argmax → Clean Token IDs (B, T)
    
    關鍵: Codebook 完全凍結，不訓練任何 embedding
    """
    
    def __init__(
        self,
        codebook,           # (4096, 512) WavTokenizer 的 Codebook
        d_model=512,        # Transformer 維度
        nhead=8,            # Multi-head 數量
        num_layers=6,       # Transformer 層數
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=5000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = codebook.shape[0]  # 4096
        
        # Frozen Codebook as Embedding Layer
        self.register_buffer('codebook', codebook)
        # 不需要 nn.Embedding，直接用 codebook 查表
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output Projection to Vocabulary
        self.output_proj = nn.Linear(d_model, self.vocab_size)
        
    def forward(self, noisy_token_ids, return_logits=False):
        """
        Args:
            noisy_token_ids: (B, T) Noisy Token IDs [0, 4095]
            return_logits: 是否返回 logits (訓練時用)
        
        Returns:
            clean_token_ids: (B, T) Predicted Clean Token IDs
            或
            logits: (B, T, 4096) 如果 return_logits=True
        """
        B, T = noisy_token_ids.shape
        
        # Step 1: Frozen Codebook Lookup
        # 完全不訓練 embedding，直接從 WavTokenizer Codebook 查表
        embeddings = self.codebook[noisy_token_ids]  # (B, T, 512)
        
        # Step 2: Positional Encoding
        embeddings = self.pos_encoding(embeddings)  # (B, T, 512)
        
        # Step 3: Transformer Encoding
        hidden = self.transformer_encoder(embeddings)  # (B, T, 512)
        
        # Step 4: Project to Vocabulary
        logits = self.output_proj(hidden)  # (B, T, 4096)
        
        if return_logits:
            return logits
        else:
            # Greedy Decoding
            clean_token_ids = logits.argmax(dim=-1)  # (B, T)
            return clean_token_ids


class PositionalEncoding(nn.Module):
    """標準的 Sinusoidal Positional Encoding"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 預計算 positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
"""
離散Token專用Transformer架構優化

解決transformer與離散化不相容的問題
主要改進：
1. 離散感知的位置編碼
2. 改進的注意力機制
3. 局部性增強策略
4. 梯度流優化

實驗背景：
- 注意力熵退化: -4.605
- 局部性損失: -0.000  
- 平滑度下降: 0.020
- 位置內容衝突: 0.227

作者：AI Research Assistant
日期：2025-10-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

class DiscreteAwarePositionalEncoding(nn.Module):
    """離散感知的位置編碼
    
    專門為離散token設計，解決位置編碼與量化值衝突的問題
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        discrete_vocab_size: int = 4096,
        alpha: float = 0.1
    ):
        """初始化離散感知位置編碼
        
        Args:
            d_model: 模型維度
            max_len: 最大序列長度
            discrete_vocab_size: 離散詞彙表大小
            alpha: 離散調制強度
        """
        super().__init__()
        
        self.d_model = d_model
        self.alpha = alpha
        
        # 標準正弦位置編碼
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # 離散token位置調制
        self.discrete_modulation = nn.Embedding(discrete_vocab_size, d_model)
        nn.init.normal_(self.discrete_modulation.weight, std=0.02)
        
        # 位置-內容交互層
        self.position_content_fusion = nn.Linear(d_model * 2, d_model)
    
    def forward(self, token_embeddings: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """前向傳播
        
        Args:
            token_embeddings: [batch, seq_len, d_model] token嵌入
            token_ids: [batch, seq_len] token ID
            
        Returns:
            位置編碼後的嵌入
        """
        seq_len = token_embeddings.size(1)
        
        # 標準位置編碼
        pos_encoding = self.pe[:, :seq_len]
        
        # 離散token特定的位置調制
        discrete_pos = self.discrete_modulation(token_ids)
        
        # 融合位置信息和token信息
        combined = torch.cat([
            token_embeddings + pos_encoding,
            token_embeddings + self.alpha * discrete_pos
        ], dim=-1)
        
        fused = self.position_content_fusion(combined)
        
        return fused

class LocalityEnhancedAttention(nn.Module):
    """局部性增強的注意力機制
    
    專門為離散token設計，增強局部注意力模式
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        local_window: int = 16,
        locality_strength: float = 2.0,
        dropout: float = 0.1
    ):
        """初始化局部性增強注意力
        
        Args:
            d_model: 模型維度
            num_heads: 注意力頭數
            local_window: 局部窗口大小
            locality_strength: 局部性增強強度
            dropout: dropout率
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.local_window = local_window
        self.locality_strength = locality_strength
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 局部性偏置
        self.local_bias = nn.Parameter(torch.zeros(1, 1, local_window * 2 + 1))
        
        # 初始化
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向傳播
        
        Args:
            query, key, value: [batch, seq_len, d_model]
            mask: 注意力遮罩
            
        Returns:
            注意力輸出
        """
        batch_size, seq_len, _ = query.size()
        
        # 線性變換
        q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 計算注意力分數
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 添加局部性偏置
        if seq_len <= self.local_window * 2 + 1:
            # 短序列：直接使用全部偏置
            local_bias = self.local_bias[:, :, :seq_len]
        else:
            # 長序列：為每個位置創建局部偏置
            local_bias = self._create_local_bias(seq_len, scores.device)
        
        scores = scores + self.locality_strength * local_bias
        
        # 應用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 應用注意力
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(output)
    
    def _create_local_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """為長序列創建局部偏置矩陣"""
        bias_matrix = torch.zeros(1, 1, seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            start = max(0, i - self.local_window)
            end = min(seq_len, i + self.local_window + 1)
            
            # 計算相對位置
            relative_positions = torch.arange(start, end, device=device) - i
            bias_indices = relative_positions + self.local_window
            
            # 應用偏置
            valid_indices = (bias_indices >= 0) & (bias_indices < self.local_bias.size(-1))
            if valid_indices.any():
                bias_matrix[0, 0, i, start:end][valid_indices] = \
                    self.local_bias[0, 0, bias_indices[valid_indices]]
        
        return bias_matrix

class DiscreteTokenTransformerLayer(nn.Module):
    """離散Token專用Transformer層"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        local_window: int = 16,
        use_residual_scaling: bool = True
    ):
        """初始化離散Token Transformer層
        
        Args:
            d_model: 模型維度
            num_heads: 注意力頭數
            d_ff: 前饋網路維度
            dropout: dropout率
            local_window: 局部窗口大小
            use_residual_scaling: 是否使用殘差縮放
        """
        super().__init__()
        
        self.attention = LocalityEnhancedAttention(
            d_model, num_heads, local_window, dropout=dropout
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 殘差縮放用於穩定訓練
        self.use_residual_scaling = use_residual_scaling
        if use_residual_scaling:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向傳播"""
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        attn_output = self.attention(x, x, x, mask)
        
        if self.use_residual_scaling:
            x = residual + self.residual_scale * self.dropout(attn_output)
        else:
            x = residual + self.dropout(attn_output)
        
        # Feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        
        if self.use_residual_scaling:
            x = residual + self.residual_scale * self.dropout(ff_output)
        else:
            x = residual + self.dropout(ff_output)
        
        return x

class DiscreteTokenTransformer(nn.Module):
    """離散Token專用Transformer模型"""
    
    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 1024,
        dropout: float = 0.1,
        local_window: int = 16
    ):
        """初始化離散Token Transformer
        
        Args:
            vocab_size: 詞彙表大小
            d_model: 模型維度
            num_layers: 層數
            num_heads: 注意力頭數
            d_ff: 前饋網路維度
            max_len: 最大序列長度
            dropout: dropout率
            local_window: 局部窗口大小
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置編碼
        self.pos_encoding = DiscreteAwarePositionalEncoding(
            d_model, max_len, vocab_size
        )
        
        # Transformer層
        self.layers = nn.ModuleList([
            DiscreteTokenTransformerLayer(
                d_model, num_heads, d_ff, dropout, local_window
            ) for _ in range(num_layers)
        ])
        
        # 輸出投影
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 初始化
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向傳播
        
        Args:
            input_tokens: [batch, seq_len] 輸入token
            target_tokens: [batch, seq_len] 目標token（可選）
            mask: 注意力遮罩
            
        Returns:
            logits或token嵌入
        """
        # Token嵌入
        x = self.token_embedding(input_tokens) * math.sqrt(self.d_model)
        
        # 位置編碼
        x = self.pos_encoding(x, input_tokens)
        x = self.dropout(x)
        
        # Transformer層
        for layer in self.layers:
            x = layer(x, mask)
        
        # 輸出投影
        if target_tokens is not None:
            # 訓練模式：返回logits
            return self.output_projection(x)
        else:
            # 推理模式：返回特徵
            return x

def upgrade_wavtokenizer_transformer(original_model, config=None):
    """升級WavTokenizer的Transformer為離散專用版本
    
    Args:
        original_model: 原始模型
        config: 配置參數
        
    Returns:
        升級後的模型
    """
    if config is None:
        config = {
            'vocab_size': 4096,
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'local_window': 16
        }
    
    # 創建新的離散專用transformer
    discrete_transformer = DiscreteTokenTransformer(**config)
    
    # 如果可能，轉移權重
    try:
        if hasattr(original_model, 'token_embedding'):
            discrete_transformer.token_embedding.weight.data.copy_(
                original_model.token_embedding.weight.data
            )
        
        if hasattr(original_model, 'output_projection'):
            discrete_transformer.output_projection.weight.data.copy_(
                original_model.output_projection.weight.data
            )
        
        logging.info("成功轉移部分權重到離散專用Transformer")
    except Exception as e:
        logging.warning(f"權重轉移失敗，使用隨機初始化: {e}")
    
    return discrete_transformer

if __name__ == "__main__":
    print("🤖 離散Token專用Transformer架構完成！")
    print("\n主要改進：")
    print("1. 離散感知位置編碼 - 解決位置-內容衝突")
    print("2. 局部性增強注意力 - 改善注意力模式")
    print("3. 殘差縮放 - 穩定梯度流")
    print("4. 預歸一化 - 提高訓練穩定性")
    print("\n預期效果：")
    print("- 注意力熵從-4.605改善到正常範圍")
    print("- 增強局部性，提高序列建模能力")
    print("- 減少梯度退化，穩定訓練過程")
    print("- 更好地處理離散token的特性")
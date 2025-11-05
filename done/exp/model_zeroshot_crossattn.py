"""
Zero-Shot Speaker-Conditioned Token Denoising Transformer
WITH CROSS-ATTENTION FUSION

實驗編號: EXP-20251105-CrossAttn
目的: 驗證假設 2 - Speaker Embedding 影響力不足
改進: 將 Additive Fusion 改為 Cross-Attention Mechanism

與原始版本的差異:
1. ✅ 新增 CrossAttentionFusion module
2. ✅ 改用 Cross-Attention 替代簡單相加
3. ✅ 支持返回 attention weights (用於視覺化)
4. ✅ 其餘架構保持不變（保持可比性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion for Speaker Embedding
    
    將 Speaker Embedding 通過 Cross-Attention 注入到 Token Embeddings
    
    架構:
        Token Embeddings (B, T, d_model) 作為 Query
        Speaker Embedding (B, d_model) 作為 Key & Value
        
        每個 token 動態決定需要多少 speaker 資訊
    
    參數:
        d_model: int - 模型維度 (512)
        nhead: int - 注意力頭數 (8)
        dropout: float - Dropout 比率
    
    輸入:
        token_emb: (B, T, d_model) - Token embeddings (Query)
        speaker_emb: (B, d_model) - Speaker embedding (Key & Value)
    
    輸出:
        fused_emb: (B, T, d_model) - Fusion 後的 embeddings
        attn_weights: (B, T, 1) - 每個 token 對 speaker 的 attention 分數
    """
    
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        
        # Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm (for residual connection)
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_emb, speaker_emb):
        """
        Args:
            token_emb: (B, T, d_model) - Token embeddings
            speaker_emb: (B, d_model) - Speaker embedding
        
        Returns:
            fused_emb: (B, T, d_model) - Fused embeddings
            attn_weights: (B, T, 1) - Attention weights
        """
        B, T, D = token_emb.shape
        
        # Speaker embedding: (B, d_model) → (B, 1, d_model)
        # Speaker 作為一個"全局" key/value
        speaker_kv = speaker_emb.unsqueeze(1)  # (B, 1, d_model)
        
        # Cross-Attention
        # Query: token_emb (B, T, d_model) - 每個 token 問: "我需要多少 speaker 資訊?"
        # Key:   speaker_kv (B, 1, d_model) - Speaker 提供的資訊維度
        # Value: speaker_kv (B, 1, d_model) - Speaker 提供的實際資訊
        attn_output, attn_weights = self.cross_attn(
            query=token_emb,      # (B, T, d_model)
            key=speaker_kv,       # (B, 1, d_model)
            value=speaker_kv,     # (B, 1, d_model)
            need_weights=True
        )
        # attn_output: (B, T, d_model) - 每個 token 獲得的 speaker 資訊
        # attn_weights: (B, T, 1) - 每個 token 對 speaker 的 attention 分數
        
        # Residual Connection + Dropout + Layer Norm
        # 重要: 保留原始 token embedding, 只添加 speaker 資訊
        fused_emb = self.norm(token_emb + self.dropout(attn_output))
        
        return fused_emb, attn_weights


class ZeroShotDenoisingTransformerCrossAttn(nn.Module):
    """
    Speaker-Conditioned Token Denoising Transformer
    WITH CROSS-ATTENTION FUSION

    架構:
        Noisy Tokens (B, T) + Speaker Embedding (B, D_spk)
        → Token Emb (B, T, 512)
        → Positional Encoding
        → Cross-Attention Fusion with Speaker
        → Transformer Encoder
        → Output Projection → (B, T, 4096)

    與 Additive Fusion 版本差異:
        ❌ 移除: 簡單相加 (token_emb + speaker_emb)
        ✅ 新增: CrossAttentionFusion (動態 attention)
        ✅ 優勢: 每個 token 可動態決定需要多少 speaker 資訊
    """

    def __init__(
        self,
        codebook,               # (4096, 512) WavTokenizer Codebook
        speaker_embed_dim=256,  # Speaker encoder 輸出維度
        d_model=512,            # Transformer 維度
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=5000
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = codebook.shape[0]  # 4096

        # ============================================================
        # 1. Frozen Codebook (與 Baseline 相同)
        # ============================================================
        self.register_buffer('codebook', codebook)

        # ============================================================
        # 2. Speaker Embedding Projection (與 Baseline 相同)
        # ============================================================
        self.speaker_proj = nn.Linear(speaker_embed_dim, d_model)

        # ============================================================
        # 3. Positional Encoding (與 Baseline 相同)
        # ============================================================
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        # ============================================================
        # 4. Cross-Attention Fusion (新增)
        # ============================================================
        self.cross_attn_fusion = CrossAttentionFusion(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )

        # ============================================================
        # 5. Transformer Encoder (與 Baseline 相同)
        # ============================================================
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

        # ============================================================
        # 6. Output Projection (與 Baseline 相同)
        # ============================================================
        self.output_proj = nn.Linear(d_model, self.vocab_size)

    def forward(self, noisy_token_ids, speaker_embedding, 
                return_logits=False, return_attention=False):
        """
        Args:
            noisy_token_ids: (B, T) Noisy Token IDs [0, 4095]
            speaker_embedding: (B, D_spk) Speaker embedding from speaker encoder
            return_logits: 是否返回 logits
            return_attention: 是否返回 cross-attention weights (用於視覺化)

        Returns:
            clean_token_ids: (B, T) Predicted Clean Token IDs
            或
            logits: (B, T, 4096) 如果 return_logits=True
            或
            (logits, attn_weights) 如果 return_attention=True
        """
        B, T = noisy_token_ids.shape

        # ============================================================
        # Step 1: Token Embedding (Frozen Codebook Lookup)
        # ============================================================
        token_emb = self.codebook[noisy_token_ids]  # (B, T, 512)

        # ============================================================
        # Step 2: Positional Encoding
        # ============================================================
        token_emb = self.pos_encoding(token_emb)  # (B, T, 512)

        # ============================================================
        # Step 3: Speaker Embedding Projection
        # ============================================================
        speaker_emb = self.speaker_proj(speaker_embedding)  # (B, D_spk) -> (B, 512)

        # ============================================================
        # Step 4: Cross-Attention Fusion (關鍵改進)
        # ============================================================
        fused_emb, attn_weights = self.cross_attn_fusion(
            token_emb=token_emb,      # (B, T, 512) - Query
            speaker_emb=speaker_emb   # (B, 512) - Key & Value
        )
        # fused_emb: (B, T, 512) - 融合後的 embeddings
        # attn_weights: (B, T, 1) - 每個 token 對 speaker 的 attention 分數

        # ============================================================
        # Step 5: Transformer Encoding
        # ============================================================
        hidden = self.transformer_encoder(fused_emb)  # (B, T, 512)

        # ============================================================
        # Step 6: Output Projection
        # ============================================================
        logits = self.output_proj(hidden)  # (B, T, 4096)

        if return_attention:
            return logits, attn_weights
        elif return_logits:
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


# ============================================================================
#                            測試代碼
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("測試 ZeroShotDenoisingTransformerCrossAttn")
    print("=" * 80)

    # 模擬輸入
    batch_size = 4
    seq_len = 100
    codebook_size = 4096
    codebook_dim = 512

    # 創建隨機 codebook (模擬 WavTokenizer)
    codebook = torch.randn(codebook_size, codebook_dim)

    # 創建模型
    model = ZeroShotDenoisingTransformerCrossAttn(
        codebook=codebook,
        speaker_embed_dim=256,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1
    )

    print(f"\n模型架構:")
    print(f"  - d_model: {model.d_model}")
    print(f"  - vocab_size: {model.vocab_size}")

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(b.numel() for b in model.buffers())

    print(f"\n參數統計:")
    print(f"  - 總參數: {total_params:,}")
    print(f"  - 可訓練參數: {trainable_params:,}")
    print(f"  - 凍結參數 (Codebook): {frozen_params:,}")
    
    # 計算新增參數量
    cross_attn_params = sum(p.numel() for n, p in model.named_parameters() 
                           if 'cross_attn_fusion' in n)
    print(f"  - Cross-Attention 新增參數: {cross_attn_params:,}")

    # 模擬輸入
    noisy_tokens = torch.randint(0, codebook_size, (batch_size, seq_len))
    speaker_emb = torch.randn(batch_size, 256)

    print(f"\n輸入:")
    print(f"  - Noisy tokens: {noisy_tokens.shape}")
    print(f"  - Speaker embedding: {speaker_emb.shape}")

    # Forward pass (logits)
    logits = model(noisy_tokens, speaker_emb, return_logits=True)
    print(f"\n輸出 (logits):")
    print(f"  - Shape: {logits.shape}")

    # Forward pass (predictions)
    pred_tokens = model(noisy_tokens, speaker_emb, return_logits=False)
    print(f"\n輸出 (predictions):")
    print(f"  - Shape: {pred_tokens.shape}")

    # Forward pass (with attention weights)
    logits, attn_weights = model(noisy_tokens, speaker_emb, 
                                  return_logits=True, return_attention=True)
    print(f"\n輸出 (with attention):")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Attention weights shape: {attn_weights.shape}")
    print(f"  - Attention weights range: [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
    print(f"  - Attention weights mean: {attn_weights.mean():.4f}")

    # 測試損失計算
    target_tokens = torch.randint(0, codebook_size, (batch_size, seq_len))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, codebook_size), target_tokens.view(-1))
    print(f"\n損失:")
    print(f"  - CrossEntropy Loss: {loss.item():.4f}")

    print("\n" + "=" * 80)
    print("✅ 所有測試通過！Cross-Attention 模型準備就緒")
    print("=" * 80)

"""
Zero-Shot Speaker-Conditioned Token Denoising Transformer
Cross-Attention with Learnable Gate (per-token)

目的:
- 在 Cross-Attention 殘差上加入 per-token gate，期望抑制低 margin 區的破壞性擾動、
  放大具方向性的擾動（學習式 gate 版本）。
"""

import math
import torch
import torch.nn as nn


class CrossAttentionFusionGated(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1, speaker_tokens: int = 4):
        super().__init__()
        self.speaker_tokens = speaker_tokens

        # Expand single speaker embedding to K tokens + learnable pos
        self.spk_expand = nn.Sequential(
            nn.Linear(d_model, d_model * speaker_tokens),
            nn.ReLU(inplace=True),
        )
        self.spk_pos = nn.Parameter(torch.randn(1, speaker_tokens, d_model) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Per-token gate: g = sigmoid(MLP(LN(token_emb))) in (B,T,1)
        hidden_gate = max(16, d_model // 4)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_gate),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_gate, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_emb, speaker_emb):
        B, T, D = token_emb.shape
        spk_tokens = self.spk_expand(speaker_emb).view(B, self.speaker_tokens, D) + self.spk_pos
        attn_output, attn_weights = self.cross_attn(
            query=token_emb,
            key=spk_tokens,
            value=spk_tokens,
            need_weights=True,
        )
        g = self.gate(token_emb)  # (B,T,1)
        fused = self.norm(token_emb + self.dropout(g * attn_output))
        return fused, attn_weights, g


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ZeroShotDenoisingTransformerCrossAttnGated(nn.Module):
    def __init__(
        self,
        codebook,
        speaker_embed_dim=256,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=5000,
        speaker_tokens: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = codebook.shape[0]
        self.register_buffer('codebook', codebook)

        self.speaker_proj = nn.Linear(speaker_embed_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.cross_attn_fusion = CrossAttentionFusionGated(
            d_model=d_model, nhead=nhead, dropout=dropout, speaker_tokens=speaker_tokens
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, self.vocab_size)

    def forward(self, noisy_token_ids, speaker_embedding, return_logits=False, return_attention=False):
        B, T = noisy_token_ids.shape
        token_emb = self.codebook[noisy_token_ids]
        token_emb = self.pos_encoding(token_emb)
        speaker_emb = self.speaker_proj(speaker_embedding)
        fused_emb, attn_weights, gate = self.cross_attn_fusion(token_emb, speaker_emb)
        hidden = self.transformer_encoder(fused_emb)
        logits = self.output_proj(hidden)
        if return_attention:
            return logits, attn_weights, gate
        if return_logits:
            return logits
        return logits.argmax(dim=-1)

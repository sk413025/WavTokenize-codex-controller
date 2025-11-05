"""
Zero-Shot Speaker-Conditioned Transformer
Cross-Attention injected at multiple encoder layers (deep conditioning)
"""

import math
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1, speaker_tokens: int = 4):
        super().__init__()
        self.speaker_tokens = speaker_tokens
        self.spk_expand = nn.Sequential(
            nn.Linear(d_model, d_model * speaker_tokens),
            nn.ReLU(inplace=True),
        )
        self.spk_pos = nn.Parameter(torch.randn(1, speaker_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_emb, speaker_emb):
        B, T, D = token_emb.shape
        spk_tokens = self.spk_expand(speaker_emb).view(B, self.speaker_tokens, D) + self.spk_pos
        attn_output, attn_weights = self.cross_attn(
            query=token_emb, key=spk_tokens, value=spk_tokens, need_weights=True
        )
        fused = self.norm(token_emb + self.dropout(attn_output))
        return fused, attn_weights


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


class ZeroShotDenoisingTransformerCrossAttnDeep(nn.Module):
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
        inject_layers: tuple = (1, 3),
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = codebook.shape[0]
        self.register_buffer('codebook', codebook)
        self.inject_layers = set(inject_layers)

        self.speaker_proj = nn.Linear(speaker_embed_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.fusion0 = CrossAttentionFusion(d_model=d_model, nhead=nhead, dropout=dropout, speaker_tokens=speaker_tokens)
        # encoder layers as ModuleList for manual iterate
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        # extra fusion modules for deep injection (shared or separate)
        self.deep_fusion = CrossAttentionFusion(d_model=d_model, nhead=nhead, dropout=dropout, speaker_tokens=speaker_tokens)
        self.output_proj = nn.Linear(d_model, self.vocab_size)

    def forward(self, noisy_token_ids, speaker_embedding, return_logits=False, return_attention=False):
        B, T = noisy_token_ids.shape
        token_emb = self.codebook[noisy_token_ids]
        token_emb = self.pos_encoding(token_emb)
        speaker_emb = self.speaker_proj(speaker_embedding)
        hidden, attn_w_all = self.fusion0(token_emb, speaker_emb)

        for i, layer in enumerate(self.layers):
            if i in self.inject_layers:
                hidden, _ = self.deep_fusion(hidden, speaker_emb)
            hidden = layer(hidden)

        logits = self.output_proj(hidden)
        if return_logits:
            return logits
        return logits.argmax(dim=-1)


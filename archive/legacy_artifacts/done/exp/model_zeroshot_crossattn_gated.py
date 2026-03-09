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

    def forward(self, token_emb, speaker_emb, g_override=None, return_fusion_vectors: bool = False):
        B, T, D = token_emb.shape
        spk_tokens = self.spk_expand(speaker_emb).view(B, self.speaker_tokens, D) + self.spk_pos
        attn_output, attn_weights = self.cross_attn(
            query=token_emb,
            key=spk_tokens,
            value=spk_tokens,
            need_weights=True,
        )
        if g_override is None:
            g_used = self.gate(token_emb)  # (B,T,1)
        else:
            g_used = g_override
        fused = self.norm(token_emb + self.dropout(g_used * attn_output))
        if return_fusion_vectors:
            return fused, attn_weights, g_used, attn_output, token_emb
        return fused, attn_weights, g_used


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

    def forward(
        self,
        noisy_token_ids,
        speaker_embedding,
        return_logits=False,
        return_attention=False,
        return_fusion=False,
        labels=None,
        margin_cfg=None,
        g_override=None,
    ):
        B, T = noisy_token_ids.shape
        token_emb = self.codebook[noisy_token_ids]
        token_emb = self.pos_encoding(token_emb)
        speaker_emb = self.speaker_proj(speaker_embedding)

        # Optional margin-aware gating schedule
        g_sched = None
        if g_override is not None:
            g_sched = g_override
        elif margin_cfg is not None:
            # Base gate from token embedding
            with torch.no_grad():
                g_base = self.cross_attn_fusion.gate(token_emb)  # (B,T,1)
            # Prefusion logits proxy from token-only path (detach to avoid gradients)
            with torch.no_grad():
                pre_logits = self.output_proj(token_emb.detach())  # (B,T,V)
                probs = torch.softmax(pre_logits, dim=-1)
                if labels is not None:
                    # target prob and competitor prob (exclude target)
                    tgt = labels.view(B, T, 1)
                    p_t = probs.gather(-1, tgt).squeeze(-1)  # (B,T)
                    probs_excl = probs.clone()
                    probs_excl.scatter_(-1, tgt, float('-inf'))
                    p_c2, _ = probs_excl.max(dim=-1)  # (B,T)
                else:
                    # Use top-1 and top-2 predicted probabilities
                    top2 = torch.topk(probs, k=2, dim=-1).values  # (B,T,2)
                    p_t = top2[..., 0]
                    p_c2 = top2[..., 1]
                margin = (p_t - p_c2)  # in probability space
                low_thr = float(margin_cfg.get('low_thr', 0.02))
                mid_thr = float(margin_cfg.get('mid_thr', 0.4))
                mid_amp = float(margin_cfg.get('mid_amp', 1.5))
                high_amp = float(margin_cfg.get('high_amp', 0.5))
                # Broadcast base gate to (B,T)
                g_base_bt = g_base.squeeze(-1)
                g_low = torch.zeros_like(g_base_bt)
                g_mid = torch.clamp(g_base_bt * mid_amp, 0.0, 1.0)
                g_high = torch.clamp(g_base_bt * high_amp, 0.0, 1.0)
                g_bt = torch.where(margin < low_thr, g_low,
                                   torch.where(margin < mid_thr, g_mid, g_high))
                g_sched = g_bt.unsqueeze(-1)

        if return_fusion:
            fused_emb, attn_weights, gate_used, attn_vec, token_vec = self.cross_attn_fusion(
                token_emb, speaker_emb, g_override=g_sched, return_fusion_vectors=True
            )
        else:
            fused_emb, attn_weights, gate_used = self.cross_attn_fusion(
                token_emb, speaker_emb, g_override=g_sched, return_fusion_vectors=False
            )
        hidden = self.transformer_encoder(fused_emb)
        logits = self.output_proj(hidden)
        if return_attention:
            if return_fusion:
                return logits, attn_weights, gate_used, attn_vec, token_vec
            return logits, attn_weights, gate_used
        if return_logits:
            if return_fusion:
                return logits, attn_vec, token_vec, gate_used
            return logits
        return logits.argmax(dim=-1)

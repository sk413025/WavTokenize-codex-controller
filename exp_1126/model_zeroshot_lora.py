"""
Zero-Shot Speaker-Conditioned Token Denoising Transformer

與 Baseline 的差異:
1. 接收 speaker_embedding 作為額外輸入
2. Token embedding + Speaker embedding fusion (支持四種方法)
   - 'add': 簡單加法融合 (預設) - [a] + [b] → [c]
   - 'concat': 拼接融合 - [a] + [b] → [a,b]
   - 'film': FiLM 融合 (Feature-wise Linear Modulation) - γ * [a] + β → [c]
   - 'cross_attn': Cross-Attention 融合 with Learnable Gate
3. 其餘架構保持不變（Transformer Encoder）

⭐ 新增: Learnable MLP Gate (from HARD-refine branch)
- 從 token embedding 直接學習 gate 值
- Single forward pass（無需 preliminary logits）
- 端到端訓練
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """正弦位置編碼"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 創建位置編碼矩陣
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
            x: (B, T, D)
        Returns:
            (B, T, D) with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ZeroShotDenoisingTransformer(nn.Module):
    """
    Speaker-Conditioned Token Denoising Transformer

    架構:
        Noisy Tokens (B, T) + Speaker Embedding (B, D_spk)
        → Token Emb (B, T, 512) + Speaker Emb (B, T, 512)
        → Combined Emb (B, T, 512)
        → Positional Encoding
        → Transformer Encoder
        → Output Projection → (B, T, 4096)

    與 Baseline 差異:
        ✅ 新增: Speaker embedding fusion (支持 4 種方法)
        ✅ 新增: Learnable MLP Gate (from HARD-refine)
        ✅ 新增: speaker_proj layer
        ✅ 其餘: 完全相同（保持可比性）
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
        max_seq_len=5000,
        fusion_method='add',    # 'add', 'concat', 'film', 'cross_attn'
        cross_attn_heads=4,     # Cross-Attention 的 head 數量
        speaker_tokens=4,       # Cross-Attention 的 speaker token 數量
        # ⭐ Learnable Gate 參數（from HARD-refine branch）
        use_learnable_gate=False,  # 是否啟用可學習的 gate
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = codebook.shape[0]  # 4096
        self.fusion_method = fusion_method
        self.cross_attn_heads = cross_attn_heads
        self.speaker_tokens = speaker_tokens
        self.use_learnable_gate = use_learnable_gate

        # ============================================================
        # 1. Frozen Codebook (與 Baseline 相同)
        # ============================================================
        # 添加 PAD_TOKEN embedding (全零向量，不參與訓練)
        pad_embedding = torch.zeros(1, codebook.shape[1], device=codebook.device, dtype=codebook.dtype)  # (1, 512)
        extended_codebook = torch.cat([codebook, pad_embedding], dim=0)  # (4097, 512)
        self.register_buffer('codebook', extended_codebook)

        # ============================================================
        # 2. Speaker Embedding Projection (新增)
        # ============================================================
        if fusion_method == 'add':
            # 簡單加法融合：只需要一個投影層
            self.speaker_proj = nn.Linear(speaker_embed_dim, d_model)

        elif fusion_method == 'concat':
            # 拼接融合：需要投影層將拼接後的維度壓回 d_model
            self.speaker_proj = nn.Linear(speaker_embed_dim, d_model)
            self.fusion_proj = nn.Linear(d_model * 2, d_model)

        elif fusion_method == 'film':
            # FiLM 融合：需要生成 gamma 和 beta 參數
            self.gamma_proj = nn.Linear(speaker_embed_dim, d_model)
            self.beta_proj = nn.Linear(speaker_embed_dim, d_model)

            # ⭐ Learnable Gate for FiLM (optional)
            if use_learnable_gate:
                hidden_gate = max(16, d_model // 4)  # 128 for d_model=512
                self.gate_mlp = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, hidden_gate),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_gate, 1),
                    nn.Sigmoid(),
                )

        elif fusion_method == 'cross_attn':
            # Cross-Attention 融合 with Learnable Gate
            self.speaker_proj = nn.Linear(speaker_embed_dim, d_model)

            # Expand single speaker embedding to K tokens
            self.spk_expand = nn.Sequential(
                nn.Linear(d_model, d_model * speaker_tokens),
                nn.ReLU(inplace=True),
            )
            self.spk_pos = nn.Parameter(torch.randn(1, speaker_tokens, d_model) * 0.02)

            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=cross_attn_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_attn_norm = nn.LayerNorm(d_model)
            self.cross_attn_dropout = nn.Dropout(dropout)

            # ⭐ Learnable Gate for Cross-Attention (always enabled for cross_attn)
            hidden_gate = max(16, d_model // 4)
            self.gate_mlp = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_gate),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_gate, 1),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        # ============================================================
        # 3. Positional Encoding (與 Baseline 相同)
        # ============================================================
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        # ============================================================
        # 4. Transformer Encoder (與 Baseline 相同)
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
        # 5. Output Projection (與 Baseline 相同)
        # ============================================================
        self.output_proj = nn.Linear(d_model, self.vocab_size)

    def forward(
        self,
        noisy_token_ids,
        speaker_embedding,
        return_logits=False,
        return_gate=False,
    ):
        """
        Args:
            noisy_token_ids: (B, T) 噪聲 token IDs
            speaker_embedding: (B, speaker_embed_dim) speaker embedding
            return_logits: 是否返回 logits (而非 argmax)
            return_gate: 是否返回 gate 值（僅當 use_learnable_gate=True 時有效）

        Returns:
            如果 return_logits=False: (B, T) 預測的 token IDs
            如果 return_logits=True: (B, T, vocab_size) logits
            如果 return_gate=True: 額外返回 gate 值 (B, T, 1)
        """
        B, T = noisy_token_ids.shape

        # ============================================================
        # 1. Token Embedding (從 Codebook 查表)
        # ============================================================
        token_emb = self.codebook[noisy_token_ids]  # (B, T, 512)

        # ============================================================
        # 2. Speaker Embedding Fusion (四種方法)
        # ============================================================
        gate_used = None

        if self.fusion_method == 'add':
            # 方法 1: 簡單加法
            speaker_emb = self.speaker_proj(speaker_embedding)  # (B, 512)
            speaker_emb = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)
            combined_emb = token_emb + speaker_emb

        elif self.fusion_method == 'concat':
            # 方法 2: 拼接後投影
            speaker_emb = self.speaker_proj(speaker_embedding)  # (B, 512)
            speaker_emb = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)
            concatenated = torch.cat([token_emb, speaker_emb], dim=-1)  # (B, T, 1024)
            combined_emb = self.fusion_proj(concatenated)  # (B, T, 512)

        elif self.fusion_method == 'film':
            # 方法 3: FiLM (Feature-wise Linear Modulation)
            gamma = self.gamma_proj(speaker_embedding)  # (B, 512)
            beta = self.beta_proj(speaker_embedding)    # (B, 512)

            # Broadcast to (B, T, 512)
            gamma = gamma.unsqueeze(1).expand(-1, T, -1)
            beta = beta.unsqueeze(1).expand(-1, T, -1)

            # ⭐ Apply Learnable Gate (optional)
            if self.use_learnable_gate:
                gate_used = self.gate_mlp(token_emb)  # (B, T, 1)

                # 調整 gamma 和 beta
                gamma = 1.0 + gate_used * (gamma - 1.0)
                beta = gate_used * beta

            # FiLM 調製: γ ⊙ x + β
            combined_emb = gamma * token_emb + beta

        elif self.fusion_method == 'cross_attn':
            # 方法 4: Cross-Attention with Learnable Gate
            speaker_emb = self.speaker_proj(speaker_embedding)  # (B, 512)

            # Expand speaker to multiple tokens
            spk_tokens = self.spk_expand(speaker_emb).view(B, self.speaker_tokens, self.d_model)
            spk_tokens = spk_tokens + self.spk_pos  # (B, speaker_tokens, 512)

            # Cross-Attention
            attn_output, _ = self.cross_attention(
                query=token_emb,
                key=spk_tokens,
                value=spk_tokens,
                need_weights=False
            )  # (B, T, 512)

            # ⭐ Learnable Gate (always enabled for cross_attn)
            gate_used = self.gate_mlp(token_emb)  # (B, T, 1)

            # Apply gate to attention output
            gated_attn = attn_output * gate_used

            # Residual connection with LayerNorm
            combined_emb = self.cross_attn_norm(
                token_emb + self.cross_attn_dropout(gated_attn)
            )

        # ============================================================
        # 3. Positional Encoding
        # ============================================================
        combined_emb = self.pos_encoding(combined_emb)  # (B, T, 512)

        # ============================================================
        # 4. Transformer Encoder
        # ============================================================
        hidden = self.transformer_encoder(combined_emb)  # (B, T, 512)

        # ============================================================
        # 5. Output Projection
        # ============================================================
        logits = self.output_proj(hidden)  # (B, T, 4096)

        # ============================================================
        # 6. Return
        # ============================================================
        if return_gate and gate_used is not None:
            if return_logits:
                return logits, gate_used
            else:
                return logits.argmax(dim=-1), gate_used

        if return_logits:
            return logits
        else:
            return logits.argmax(dim=-1)


def count_parameters(model):
    """計算模型可訓練參數數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

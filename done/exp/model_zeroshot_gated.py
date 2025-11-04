"""
Zero-Shot Speaker-Conditioned Token Denoising Transformer with Gated Fusion

改進版本：使用 Gated Fusion 替代 Simple Addition

核心改進：
- ❌ 舊: combined = token_emb + speaker_emb (固定權重)
- ✅ 新: combined = gate * token_emb + (1-gate) * speaker_emb (學習權重)

預期效果：
- 驗證準確率提升 1-2%
- 更好地利用 speaker 信息
- 動態調節 token vs speaker 的重要性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedFusion(nn.Module):
    """
    門控融合模組

    功能：
        - 接收 token embedding 和 speaker embedding
        - 動態學習每個位置的融合權重（gate）
        - 輸出融合後的 embedding

    原理：
        gate = sigmoid(Linear(concat(token, speaker)))
        output = gate ⊙ token + (1-gate) ⊙ speaker

    其中 gate ∈ [0,1]:
        - gate ≈ 1: 更多使用 token 信息
        - gate ≈ 0: 更多使用 speaker 信息
        - gate ≈ 0.5: 平均使用兩者
    """

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()

        # Gate network: 輸入 2*d_model，輸出 d_model 的門控值
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
            nn.Sigmoid()  # 輸出 [0, 1]
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, token_emb, speaker_emb):
        """
        Args:
            token_emb: (B, T, d_model) Token embeddings
            speaker_emb: (B, T, d_model) Speaker embeddings (已 broadcast)

        Returns:
            fused_emb: (B, T, d_model) 融合後的 embeddings
        """
        # 拼接 token 和 speaker embeddings
        concat = torch.cat([token_emb, speaker_emb], dim=-1)  # (B, T, 2*d_model)

        # 計算門控權重
        gate = self.gate_network(concat)  # (B, T, d_model), 範圍 [0, 1]

        # 門控融合
        fused = gate * token_emb + (1 - gate) * speaker_emb

        # Layer normalization
        fused = self.layer_norm(fused)

        return fused


class ZeroShotDenoisingTransformerGated(nn.Module):
    """
    Zero-Shot Speaker-Conditioned Token Denoising Transformer
    使用 Gated Fusion

    架構:
        Noisy Tokens (B, T) + Speaker Embedding (B, D_spk)
        → Token Emb (B, T, 512)
        → Speaker Emb (B, T, 512)
        → Gated Fusion (B, T, 512)  ← 核心改進
        → Positional Encoding
        → Transformer Encoder
        → Output Projection → (B, T, 4096)

    與原版差異:
        ✅ 使用 GatedFusion 替代 simple addition
        ✅ 其餘架構完全相同
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
        # 1. Frozen Codebook (與原版相同)
        # ============================================================
        self.register_buffer('codebook', codebook)

        # ============================================================
        # 2. Speaker Embedding Projection (與原版相同)
        # ============================================================
        self.speaker_proj = nn.Linear(speaker_embed_dim, d_model)

        # ============================================================
        # 3. Gated Fusion (核心改進) ⭐
        # ============================================================
        self.gated_fusion = GatedFusion(d_model=d_model, dropout=dropout)

        # ============================================================
        # 4. Positional Encoding (與原版相同)
        # ============================================================
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        # ============================================================
        # 5. Transformer Encoder (與原版相同)
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
        # 6. Output Projection (與原版相同)
        # ============================================================
        self.output_proj = nn.Linear(d_model, self.vocab_size)

    def forward(self, noisy_token_ids, speaker_embedding, return_logits=False):
        """
        Args:
            noisy_token_ids: (B, T) Noisy Token IDs [0, 4095]
            speaker_embedding: (B, D_spk) Speaker embedding from speaker encoder
            return_logits: 是否返回 logits

        Returns:
            clean_token_ids: (B, T) Predicted Clean Token IDs
            或
            logits: (B, T, 4096) 如果 return_logits=True
        """
        B, T = noisy_token_ids.shape

        # ============================================================
        # Step 1: Token Embedding (Frozen Codebook Lookup)
        # ============================================================
        token_emb = self.codebook[noisy_token_ids]  # (B, T, 512)

        # ============================================================
        # Step 2: Speaker Embedding Projection & Broadcasting
        # ============================================================
        speaker_emb = self.speaker_proj(speaker_embedding)  # (B, D_spk) -> (B, 512)
        speaker_emb = speaker_emb.unsqueeze(1)  # (B, 1, 512)
        speaker_emb = speaker_emb.expand(-1, T, -1)  # (B, T, 512)

        # ============================================================
        # Step 3: Gated Fusion ⭐ (核心改進)
        # ============================================================
        combined_emb = self.gated_fusion(token_emb, speaker_emb)  # (B, T, 512)

        # ============================================================
        # Step 4: Positional Encoding
        # ============================================================
        combined_emb = self.pos_encoding(combined_emb)  # (B, T, 512)

        # ============================================================
        # Step 5: Transformer Encoding
        # ============================================================
        hidden = self.transformer_encoder(combined_emb)  # (B, T, 512)

        # ============================================================
        # Step 6: Output Projection
        # ============================================================
        logits = self.output_proj(hidden)  # (B, T, 4096)

        if return_logits:
            return logits
        else:
            # Greedy Decoding
            clean_token_ids = logits.argmax(dim=-1)  # (B, T)
            return clean_token_ids


class PositionalEncoding(nn.Module):
    """標準的 Sinusoidal Positional Encoding (與原版相同)"""

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


# ============================================================
# 測試代碼
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Gated Fusion 模型測試")
    print("=" * 80)

    # 模擬參數
    B, T = 4, 100  # batch_size=4, sequence_length=100
    vocab_size = 4096
    d_model = 512
    speaker_dim = 256

    # 創建假的 codebook
    codebook = torch.randn(vocab_size, d_model)

    # 創建模型
    model = ZeroShotDenoisingTransformerGated(
        codebook=codebook,
        speaker_embed_dim=speaker_dim,
        d_model=d_model,
        nhead=8,
        num_layers=3,  # 使用 3 層
        dim_feedforward=2048,
        dropout=0.2
    )

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gated_fusion_params = sum(p.numel() for p in model.gated_fusion.parameters())

    print(f"\n模型參數統計:")
    print(f"  總參數: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}")
    print(f"  Gated Fusion 參數: {gated_fusion_params:,}")
    print(f"  相比原版增加: {gated_fusion_params:,} 參數 (~{gated_fusion_params/1e6:.2f}M)")

    # 測試前向傳播
    noisy_tokens = torch.randint(0, vocab_size, (B, T))
    speaker_emb = torch.randn(B, speaker_dim)

    print(f"\n測試前向傳播:")
    print(f"  輸入 noisy_tokens: {noisy_tokens.shape}")
    print(f"  輸入 speaker_emb: {speaker_emb.shape}")

    model.eval()
    with torch.no_grad():
        # 測試返回 logits
        logits = model(noisy_tokens, speaker_emb, return_logits=True)
        print(f"  輸出 logits: {logits.shape}")

        # 測試返回 predicted tokens
        pred_tokens = model(noisy_tokens, speaker_emb, return_logits=False)
        print(f"  輸出 pred_tokens: {pred_tokens.shape}")

    print(f"\n✅ 模型測試通過！")
    print("=" * 80)

    # 分析 gate 的行為
    print("\n分析 Gated Fusion 的門控行為:")
    model.eval()
    with torch.no_grad():
        # 提取中間的 gate 值（需要修改 forward 來返回）
        token_emb = model.codebook[noisy_tokens]
        speaker_emb_proj = model.speaker_proj(speaker_emb).unsqueeze(1).expand(-1, T, -1)

        concat = torch.cat([token_emb, speaker_emb_proj], dim=-1)
        gate = model.gated_fusion.gate_network(concat)

        print(f"  Gate 形狀: {gate.shape}")
        print(f"  Gate 平均值: {gate.mean().item():.4f}")
        print(f"  Gate 標準差: {gate.std().item():.4f}")
        print(f"  Gate 範圍: [{gate.min().item():.4f}, {gate.max().item():.4f}]")
        print(f"\n  解讀:")
        print(f"    - 平均值 ≈ 0.5: 平衡使用 token 和 speaker")
        print(f"    - 平均值 > 0.5: 更依賴 token embedding")
        print(f"    - 平均值 < 0.5: 更依賴 speaker embedding")

    print("=" * 80)

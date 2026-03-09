"""
Zero-Shot Speaker-Conditioned Token Denoising Transformer

與 Baseline 的差異:
1. 接收 speaker_embedding 作為額外輸入
2. Token embedding + Speaker embedding fusion
3. 其餘架構保持不變（Transformer Encoder）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        ✅ 新增: Speaker embedding fusion
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
        dropout=0.1,            # 建議使用 dropout 防止過擬合
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
        # 2. Speaker Embedding Projection (新增)
        # ============================================================
        self.speaker_proj = nn.Linear(speaker_embed_dim, d_model)

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
        # 防止 PAD_TOKEN=4096 導致 index 越界
        valid_token_ids = torch.clamp(noisy_token_ids, 0, self.vocab_size - 1)
        token_emb = self.codebook[valid_token_ids]  # (B, T, 512)

        # ============================================================
        # Step 2: Speaker Embedding Projection & Broadcasting
        # ============================================================
        speaker_emb = self.speaker_proj(speaker_embedding)  # (B, D_spk) -> (B, 512)
        speaker_emb = speaker_emb.unsqueeze(1)  # (B, 1, 512)
        speaker_emb = speaker_emb.expand(-1, T, -1)  # (B, T, 512)

        # ============================================================
        # Step 3: Fusion (Additive)
        # ============================================================
        combined_emb = token_emb + speaker_emb  # (B, T, 512)

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
    """標準的 Sinusoidal Positional Encoding (與 Baseline 相同)"""

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
    print("測試 ZeroShotDenoisingTransformer...")

    # 模擬輸入
    batch_size = 4
    seq_len = 100
    codebook_size = 4096
    codebook_dim = 512

    # 創建隨機 codebook (模擬 WavTokenizer)
    codebook = torch.randn(codebook_size, codebook_dim)

    # 創建模型
    model = ZeroShotDenoisingTransformer(
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

    # 測試損失計算
    target_tokens = torch.randint(0, codebook_size, (batch_size, seq_len))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, codebook_size), target_tokens.view(-1))
    print(f"\n損失:")
    print(f"  - CrossEntropy Loss: {loss.item():.4f}")

    print("\n✅ 所有測試通過！")

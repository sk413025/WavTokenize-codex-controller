"""
離散 Token 訓練的混合損失函數

結合 ttt2.py 的內容一致性損失理念，適配到離散 token 訓練：

階段 1 (Early epochs): 學習內容一致性
  - 相同 content_id 的 tokens 應該有相似的 embedding
  - 讓模型學會：不同語者/材質說同一句話 → 相似的語義表示

階段 2 (Later epochs): 學習去噪重建
  - Token-level: CrossEntropy (預測正確的 clean token)
  - Embedding-level: L2 Distance (在 embedding 空間接近 clean)
  - Perceptual-level: Spectral Loss (頻譜相似度)

使用方式:
    from discrete_hybrid_loss import DiscreteHybridLoss
    
    criterion = DiscreteHybridLoss(
        codebook=codebook_weights,
        wavtokenizer=wavtokenizer,
        device=device,
        ce_weight=1.0,
        content_weight=0.5,
        embed_weight=0.3,
        spectral_weight=0.1
    )
    
    loss = criterion(
        pred_logits=logits,          # (B, T, 4096)
        target_tokens=clean_tokens,  # (B, T)
        noisy_tokens=noisy_tokens,   # (B, T)
        content_ids=content_ids,     # (B,)
        current_epoch=epoch,
        total_epochs=600
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiscreteHybridLoss(nn.Module):
    """
    離散 Token 訓練的混合損失函數
    
    結合四種損失：
    1. Token CrossEntropy: 預測正確的 token ID
    2. Content Consistency: 相同內容的 token embeddings 應該相似
    3. Embedding L2: 在 embedding 空間接近 clean token
    4. Spectral Loss: 重建音頻的頻譜相似度
    """
    
    def __init__(
        self,
        codebook,           # (4096, 512) Frozen Codebook
        wavtokenizer,       # WavTokenizer 用於解碼音頻
        device='cuda',
        ce_weight=1.0,      # CrossEntropy 權重
        content_weight=0.5, # 內容一致性權重
        embed_weight=0.3,   # Embedding L2 權重
        spectral_weight=0.1,# Spectral Loss 權重
        warmup_epochs=50    # 前 N 個 epochs 強調內容一致性
    ):
        super().__init__()
        
        self.codebook = codebook
        self.wavtokenizer = wavtokenizer
        self.device = device
        
        # 損失權重
        self.ce_weight = ce_weight
        self.content_weight_max = content_weight
        self.embed_weight = embed_weight
        self.spectral_weight = spectral_weight
        self.warmup_epochs = warmup_epochs
        
        # CrossEntropy Loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(
        self,
        pred_logits,      # (B, T, 4096) 模型預測的 logits
        target_tokens,    # (B, T) Ground truth clean tokens
        noisy_tokens,     # (B, T) Input noisy tokens
        content_ids,      # (B,) 內容 ID
        current_epoch=0,
        total_epochs=600
    ):
        """
        計算混合損失
        
        Args:
            pred_logits: 模型預測 (B, T, 4096)
            target_tokens: 目標 tokens (B, T)
            noisy_tokens: 輸入 tokens (B, T)
            content_ids: 內容 ID (B,)
            current_epoch: 當前 epoch
            total_epochs: 總 epochs
            
        Returns:
            dict: {
                'total_loss': 總損失,
                'ce_loss': CrossEntropy 損失,
                'content_loss': 內容一致性損失,
                'embed_loss': Embedding L2 損失,
                'spectral_loss': 頻譜損失 (如果計算)
            }
        """
        
        B, T, vocab_size = pred_logits.shape
        
        # ============================================================
        # 1. Token CrossEntropy Loss
        # ============================================================
        # Reshape: (B, T, 4096) -> (B*T, 4096)
        logits_flat = pred_logits.reshape(-1, vocab_size)
        target_flat = target_tokens.reshape(-1).long()
        
        ce_loss = self.ce_loss(logits_flat, target_flat)
        
        # ============================================================
        # 2. Content Consistency Loss (早期階段重要)
        # ============================================================
        # 計算動態權重：前期高，後期低
        content_weight = self._compute_content_weight(current_epoch, total_epochs)

        # 只在權重非零時計算，避免不必要的運算
        if self.content_weight_max > 0 and content_weight > 0:
            content_loss = self._compute_content_consistency_loss(
                pred_logits, noisy_tokens, content_ids
            )
        else:
            content_loss = torch.tensor(0.0, device=self.device)

        # ============================================================
        # 3. Embedding L2 Loss
        # ============================================================
        # 只在權重非零時計算，避免不必要的運算
        if self.embed_weight > 0:
            # 預測的 token IDs
            pred_tokens = pred_logits.argmax(dim=-1)  # (B, T)

            # 從 codebook 查表得到 embeddings
            pred_embeddings = self.codebook[pred_tokens]      # (B, T, 512)
            target_embeddings = self.codebook[target_tokens]  # (B, T, 512)

            # L2 距離
            embed_loss = F.mse_loss(pred_embeddings, target_embeddings)
        else:
            embed_loss = torch.tensor(0.0, device=self.device)
        
        # ============================================================
        # 4. Spectral Loss (可選，計算成本高)
        # ============================================================
        # 注意：這需要解碼 tokens 為音頻，計算成本較高
        # 可以只在每 N 個 batch 計算一次
        spectral_loss = torch.tensor(0.0, device=self.device)
        
        # 如果想啟用，可以解開註釋
        # spectral_loss = self._compute_spectral_loss(
        #     pred_tokens, target_tokens
        # )
        
        # ============================================================
        # 總損失
        # ============================================================
        total_loss = (
            self.ce_weight * ce_loss +
            content_weight * content_loss +
            self.embed_weight * embed_loss +
            self.spectral_weight * spectral_loss
        )
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss.item(),
            'content_loss': content_loss.item(),
            'embed_loss': embed_loss.item(),
            'spectral_loss': spectral_loss.item(),
            'content_weight': content_weight
        }
    
    def _compute_content_weight(self, current_epoch, total_epochs):
        """
        計算內容一致性損失的動態權重
        
        策略：指數衰減
        - Epoch 0-50: 權重從 max → max/2
        - Epoch 50+: 權重繼續衰減到接近 0
        
        類比 ttt2.py 的理念：
        - 早期：學習"相同內容應該有相似表示"
        - 後期：專注於去噪和重建
        """
        if current_epoch < self.warmup_epochs:
            # 線性衰減
            progress = current_epoch / self.warmup_epochs
            weight = self.content_weight_max * (1.0 - 0.5 * progress)
        else:
            # 指數衰減
            progress = (current_epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            weight = self.content_weight_max * 0.5 * np.exp(-3 * progress)
        
        return weight
    
    def _compute_content_consistency_loss(self, pred_logits, noisy_tokens, content_ids):
        """
        內容一致性損失：相同 content_id 的 token embeddings 應該相似
        
        核心理念 (from ttt2.py):
        - 不同語者/材質說同一句話 (content_id 相同)
        - 應該有相似的語義表示
        - 在 embedding 空間中應該接近
        
        實現方式:
        1. 獲取預測的 token embeddings
        2. 計算相同 content_id 的 embeddings 的中心
        3. 最小化每個 embedding 到其對應中心的距離
        
        Args:
            pred_logits: (B, T, 4096)
            noisy_tokens: (B, T) 用於獲取當前的 embeddings
            content_ids: (B,)
            
        Returns:
            content_loss: scalar
        """
        B, T, vocab_size = pred_logits.shape
        
        # 獲取預測的 token IDs
        pred_tokens = pred_logits.argmax(dim=-1)  # (B, T)
        
        # 從 codebook 查表得到 embeddings
        pred_embeddings = self.codebook[pred_tokens]  # (B, T, 512)
        
        # 平均池化到句子級別 (B, 512)
        sentence_embeddings = pred_embeddings.mean(dim=1)
        
        # 計算每個 content_id 的中心
        unique_contents = torch.unique(content_ids)
        content_loss = torch.tensor(0.0, device=self.device)
        
        for content_id in unique_contents:
            # 找到相同 content_id 的所有樣本
            mask = (content_ids == content_id)
            if mask.sum() <= 1:
                continue  # 至少需要 2 個樣本才能計算一致性
            
            # 獲取這些樣本的 embeddings
            content_embeddings = sentence_embeddings[mask]  # (N, 512)
            
            # 計算中心
            center = content_embeddings.mean(dim=0, keepdim=True)  # (1, 512)
            
            # 計算每個樣本到中心的距離（餘弦相似度）
            similarities = F.cosine_similarity(content_embeddings, center, dim=1)
            
            # 損失：1 - 平均相似度 (越相似越好)
            content_loss += (1.0 - similarities.mean())
        
        # 平均到所有 content_id
        if len(unique_contents) > 0:
            content_loss /= len(unique_contents)
        
        return content_loss
    
    def _compute_spectral_loss(self, pred_tokens, target_tokens):
        """
        頻譜損失：比較重建音頻的頻譜
        
        注意：這需要解碼 tokens 為音頻，計算成本很高！
        建議只在驗證時或每 N 個 batch 計算一次
        
        Args:
            pred_tokens: (B, T)
            target_tokens: (B, T)
            
        Returns:
            spectral_loss: scalar
        """
        # 擴展 tokens 到 (num_quantizers, B, T) 格式
        # 只使用第一層量化器
        pred_tokens_expanded = pred_tokens.unsqueeze(0)    # (1, B, T)
        target_tokens_expanded = target_tokens.unsqueeze(0)  # (1, B, T)
        
        with torch.no_grad():
            # 解碼為音頻
            pred_features = self.wavtokenizer.codes_to_features(pred_tokens_expanded)
            target_features = self.wavtokenizer.codes_to_features(target_tokens_expanded)
            
            pred_audio = self.wavtokenizer.decode(pred_features)
            target_audio = self.wavtokenizer.decode(target_features)
        
        # 計算 STFT
        pred_spec = torch.stft(
            pred_audio.squeeze(1),
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            return_complex=True
        )
        target_spec = torch.stft(
            target_audio.squeeze(1),
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            return_complex=True
        )
        
        # L1 Loss on magnitude
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)
        
        spectral_loss = F.l1_loss(pred_mag, target_mag)
        
        return spectral_loss


# ============================================================
# 使用範例
# ============================================================

def example_usage():
    """
    使用範例
    """
    # 假設的參數
    device = 'cuda'
    batch_size = 8
    seq_len = 200
    vocab_size = 4096
    embedding_dim = 512
    
    # 創建假數據
    codebook = torch.randn(vocab_size, embedding_dim).to(device)
    pred_logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    noisy_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    content_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(device)  # 4 個不同內容，每個 2 個語者
    
    # 創建損失函數 (不使用 spectral loss 以節省計算)
    criterion = DiscreteHybridLoss(
        codebook=codebook,
        wavtokenizer=None,  # 如果不用 spectral loss 可以為 None
        device=device,
        ce_weight=1.0,
        content_weight=0.5,
        embed_weight=0.3,
        spectral_weight=0.0,  # 關閉 spectral loss
        warmup_epochs=50
    )
    
    # 計算損失
    loss_dict = criterion(
        pred_logits=pred_logits,
        target_tokens=target_tokens,
        noisy_tokens=noisy_tokens,
        content_ids=content_ids,
        current_epoch=10,
        total_epochs=600
    )
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value}")
    
    # 使用總損失進行反向傳播
    total_loss = loss_dict['total_loss']
    # total_loss.backward()


if __name__ == '__main__':
    example_usage()

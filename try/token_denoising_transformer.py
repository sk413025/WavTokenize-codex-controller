"""
Token Denoising Transformer

核心想法: 
  Noisy Token IDs → Frozen Codebook Lookup → Transformer → Output Token IDs

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


# ============================================================================
#                    Complete Denoising Pipeline
# ============================================================================

class WavTokenizerTransformerDenoiser:
    """
    完整的 Token Denoising 流程 (基於 Frozen Codebook)
    
    使用方式:
        denoiser = WavTokenizerTransformerDenoiser(
            wavtokenizer_config, 
            transformer_model_path
        )
        denoiser.denoise('noisy.wav', 'denoised.wav')
    """
    
    def __init__(self, wavtokenizer_config_path, transformer_model_path=None, device='cuda'):
        """
        初始化
        
        Args:
            wavtokenizer_config_path: WavTokenizer 配置檔路徑
            transformer_model_path: 訓練好的 Transformer 權重路徑
            device: 設備
        """
        from decoder.pretrained import WavTokenizer
        
        self.device = device
        
        # 載入 WavTokenizer (凍結)
        print("載入 WavTokenizer...")
        self.wavtokenizer = WavTokenizer.from_hparams0802(wavtokenizer_config_path)
        self.wavtokenizer.eval()
        self.wavtokenizer.to(device)
        
        # 凍結參數
        for param in self.wavtokenizer.parameters():
            param.requires_grad = False
        
        # 獲取 Codebook
        self.codebook = self.wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook
        print(f"Codebook 形狀: {self.codebook.shape}")  # (4096, 512)
        
        # 載入 Token Denoising Transformer
        print("載入 Token Denoising Transformer...")
        self.transformer = TokenDenoisingTransformer(
            codebook=self.codebook,
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        if transformer_model_path and Path(transformer_model_path).exists():
            self.transformer.load_state_dict(torch.load(transformer_model_path, map_location=device))
            print(f"✓ 從 {transformer_model_path} 載入權重")
        else:
            print("⚠ 使用未訓練的模型（需要先訓練）")
        
        self.transformer.eval()
        self.transformer.to(device)
    
    @torch.no_grad()
    def denoise(self, noisy_audio_path, output_path=None):
        """
        降噪音訊
        
        Args:
            noisy_audio_path: 噪音音檔路徑
            output_path: 輸出路徑 (可選)
        
        Returns:
            denoised_audio: (1, T) 降噪後的音訊
        """
        print(f"\n降噪: {noisy_audio_path}")
        
        # Step 1: 載入音訊
        noisy_audio, sr = torchaudio.load(noisy_audio_path)
        noisy_audio = noisy_audio.to(self.device)
        
        print(f"  音訊長度: {noisy_audio.shape[1] / 24000:.2f} 秒")
        
        # Step 2: Encode 到 Token IDs
        print("  [1/3] Encoding to Noisy Token IDs...")
        _, noisy_token_ids = self.wavtokenizer.encode_infer(
            noisy_audio, bandwidth_id=torch.tensor([0])
        )
        noisy_token_ids = noisy_token_ids[0]  # (1, T)
        print(f"    Noisy Token IDs: {noisy_token_ids.shape}")
        print(f"    Token 範圍: [{noisy_token_ids.min()}, {noisy_token_ids.max()}]")
        
        # Step 3: Transformer Denoising
        print("  [2/3] Transformer Denoising (Token → Token)...")
        clean_token_ids = self.transformer(noisy_token_ids)  # (1, T)
        print(f"    Clean Token IDs: {clean_token_ids.shape}")
        print(f"    Token 變化率: {(noisy_token_ids != clean_token_ids).float().mean().item() * 100:.2f}%")
        
        # Step 4: Decode 到音訊
        # 需要從 Token IDs 轉回 Codebook Embeddings
        print("  [3/3] Decoding to Audio...")
        clean_embeddings = self.codebook[clean_token_ids]  # (1, T, 512)
        clean_embeddings = clean_embeddings.permute(0, 2, 1)  # (1, 512, T)
        
        denoised_audio = self.wavtokenizer.decode(
            clean_embeddings, bandwidth_id=torch.tensor([0])
        )
        print(f"    Denoised Audio: {denoised_audio.shape}")
        
        # 保存
        if output_path:
            denoised_audio_cpu = denoised_audio.cpu()
            torchaudio.save(output_path, denoised_audio_cpu, sample_rate=24000)
            print(f"✓ 保存至: {output_path}")
        
        return denoised_audio.cpu()
    
    def compare_tokens(self, noisy_audio_path, clean_audio_path=None):
        """
        比較 Noisy 和 Clean 的 Token
        用於分析降噪效果
        """
        print("\n" + "="*80)
        print("Token 比較分析")
        print("="*80)
        
        # 載入並編碼 Noisy
        noisy_audio, _ = torchaudio.load(noisy_audio_path)
        noisy_audio = noisy_audio.to(self.device)
        
        with torch.no_grad():
            _, noisy_tok = self.wavtokenizer.encode_infer(
                noisy_audio, bandwidth_id=torch.tensor([0])
            )
            noisy_tok = noisy_tok[0]  # (1, T)
            
            # 降噪
            clean_tok = self.transformer(noisy_tok)
        
        print(f"\nNoisy Tokens (前20個): {noisy_tok[0, :20].cpu().numpy()}")
        print(f"Clean Tokens (前20個): {clean_tok[0, :20].cpu().numpy()}")
        
        # Token 變化率
        token_changed = (noisy_tok != clean_tok).float().mean().item()
        print(f"\nToken 變化率: {token_changed * 100:.2f}%")
        
        # 如果有真實的 Clean audio
        if clean_audio_path:
            clean_audio_gt, _ = torchaudio.load(clean_audio_path)
            clean_audio_gt = clean_audio_gt.to(self.device)
            
            with torch.no_grad():
                _, clean_tok_gt = self.wavtokenizer.encode_infer(
                    clean_audio_gt, bandwidth_id=torch.tensor([0])
                )
                clean_tok_gt = clean_tok_gt[0]
            
            # Token 準確率
            token_acc = (clean_tok == clean_tok_gt).float().mean().item()
            print(f"\nToken 準確率 (vs Ground Truth): {token_acc * 100:.2f}%")
        
        print("="*80)


# ============================================================================
#                    Training
# ============================================================================

class TokenDenoisingTrainer:
    """
    訓練 Token Denoising Transformer
    
    關鍵: Codebook 完全凍結，只訓練 Transformer 參數
    """
    
    def __init__(
        self,
        transformer,
        wavtokenizer,
        train_loader,
        val_loader=None,
        lr=1e-4,
        device='cuda'
    ):
        self.transformer = transformer.to(device)
        self.wavtokenizer = wavtokenizer.to(device).eval()
        self.device = device
        
        # 凍結 WavTokenizer
        for param in self.wavtokenizer.parameters():
            param.requires_grad = False
        
        # 確認 Codebook 也凍結
        assert not self.transformer.codebook.requires_grad, "Codebook 必須凍結！"
        
        # 優化器 (只訓練 Transformer + Output Projection)
        self.optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 學習率調度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
    def compute_loss(self, logits, target_token_ids):
        """
        計算 Cross-Entropy Loss
        
        Args:
            logits: (B, T, 4096) Transformer 輸出
            target_token_ids: (B, T) Ground Truth Clean Token IDs
        """
        # Cross-Entropy
        B, T, vocab_size = logits.shape
        
        loss = F.cross_entropy(
            logits.reshape(B * T, vocab_size),
            target_token_ids.reshape(B * T),
            ignore_index=-100  # 如果有 padding
        )
        
        # Token 準確率
        pred_tokens = logits.argmax(dim=-1)
        acc = (pred_tokens == target_token_ids).float().mean()
        
        return loss, acc.item()
    
    def train_epoch(self):
        """訓練一個 epoch"""
        self.transformer.train()
        
        total_loss = 0
        total_acc = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch in pbar:
            noisy_audio = batch['noisy'].to(self.device)
            clean_audio = batch['clean'].to(self.device)
            
            # Encode to Token IDs (凍結)
            with torch.no_grad():
                _, noisy_token_ids = self.wavtokenizer.encode_infer(
                    noisy_audio, bandwidth_id=torch.tensor([0])
                )
                _, clean_token_ids = self.wavtokenizer.encode_infer(
                    clean_audio, bandwidth_id=torch.tensor([0])
                )
                
                noisy_token_ids = noisy_token_ids[0]  # (B, T)
                clean_token_ids = clean_token_ids[0]  # (B, T)
            
            # Transformer Forward
            logits = self.transformer(noisy_token_ids, return_logits=True)
            
            # Loss
            loss, acc = self.compute_loss(logits, clean_token_ids)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            total_acc += acc
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc * 100:.2f}%'
            })
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self):
        """驗證"""
        if self.val_loader is None:
            return None, None
        
        self.transformer.eval()
        
        total_loss = 0
        total_acc = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            noisy_audio = batch['noisy'].to(self.device)
            clean_audio = batch['clean'].to(self.device)
            
            _, noisy_token_ids = self.wavtokenizer.encode_infer(
                noisy_audio, bandwidth_id=torch.tensor([0])
            )
            _, clean_token_ids = self.wavtokenizer.encode_infer(
                clean_audio, bandwidth_id=torch.tensor([0])
            )
            
            noisy_token_ids = noisy_token_ids[0]
            clean_token_ids = clean_token_ids[0]
            
            logits = self.transformer(noisy_token_ids, return_logits=True)
            
            loss, acc = self.compute_loss(logits, clean_token_ids)
            
            total_loss += loss.item()
            total_acc += acc
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)
        
        return avg_loss, avg_acc


# ============================================================================
#                    Main Usage Example
# ============================================================================

if __name__ == "__main__":
    print("Token Denoising Transformer (Frozen Codebook) - 使用範例")
    print("="*80)
    
    # 配置
    wavtokenizer_config = "./config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    transformer_model_path = "./checkpoints/token_denoising_transformer_best.pt"
    
    # 創建 Denoiser
    denoiser = WavTokenizerTransformerDenoiser(
        wavtokenizer_config,
        transformer_model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 降噪
    print("\n使用範例:")
    print("  denoiser.denoise('noisy.wav', 'denoised.wav')")
    print("  denoiser.compare_tokens('noisy.wav', 'clean_gt.wav')")
    
    print("\n架構說明:")
    print("  ✓ Codebook 完全凍結 (直接使用 WavTokenizer 的)")
    print("  ✓ 只訓練 Transformer + Output Projection")
    print("  ✓ 輸入/輸出都是 Token IDs")
    print("  ✓ 類比機器翻譯的 Seq2Seq")

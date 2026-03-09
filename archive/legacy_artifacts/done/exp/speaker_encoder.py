"""
Noise-Robust Speaker Encoder for Zero-Shot Denoising

提供兩種 Speaker Encoder:
1. PretrainedSpeakerEncoder: 使用預訓練模型 (resemblyzer 或 ECAPA-TDNN)
2. SimpleS peakerEncoder: 簡單的 CNN-based encoder (可選)

推薦使用選項 1 (預訓練模型) 以獲得更好的泛化能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np


# ============================================================================
#                    選項 1: 使用預訓練模型 (推薦)
# ============================================================================

class PretrainedSpeakerEncoder(nn.Module):
    """
    使用預訓練的 Speaker Encoder

    支持:
    1. Resemblyzer (簡單易用)
    2. SpeechBrain ECAPA-TDNN (效果更好)

    Args:
        model_type: 'resemblyzer' 或 'ecapa'
        freeze: 是否凍結 encoder 參數
        output_dim: 輸出維度（會自動 project）
    """

    def __init__(self, model_type='resemblyzer', freeze=True, output_dim=256):
        super().__init__()

        self.model_type = model_type
        self.freeze = freeze
        self.output_dim = output_dim

        if model_type == 'resemblyzer':
            self._init_resemblyzer()
        elif model_type == 'ecapa':
            self._init_ecapa()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Freeze parameters if needed
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def _init_resemblyzer(self):
        """初始化 Resemblyzer encoder"""
        try:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder()
            self.embed_dim = 256  # Resemblyzer 輸出 256-dim

            # Projection layer (如果需要不同的輸出維度)
            if self.output_dim != self.embed_dim:
                self.proj = nn.Linear(self.embed_dim, self.output_dim)
            else:
                self.proj = nn.Identity()

            print(f"✅ Loaded Resemblyzer encoder (output_dim={self.output_dim})")

        except ImportError:
            print("❌ Resemblyzer not installed. Install with: pip install resemblyzer")
            raise

    def _init_ecapa(self):
        """初始化 ECAPA-TDNN encoder"""
        import os
        try:
            # Try new import path first (SpeechBrain 1.0+)
            try:
                from speechbrain.inference.interfaces import EncoderClassifier
            except ImportError:
                # Fallback to old import path
                from speechbrain.pretrained import EncoderClassifier

            # 優先使用本地已下載的模型（如果存在）
            local_model_path = "pretrained_models/spkrec-ecapa-voxceleb"

            # 檢查本地模型是否存在
            if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "hyperparams.yaml")):
                print(f"✓ 使用本地 ECAPA-TDNN 模型: {local_model_path}")
                # 直接從本地目錄載入（使用原始方式，不指定 run_opts）
                self.encoder = EncoderClassifier.from_hparams(
                    source=local_model_path,
                    savedir=local_model_path
                )
            else:
                print(f"✓ 從 HuggingFace 下載 ECAPA-TDNN 模型...")
                # 從 HuggingFace 下載
                self.encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=local_model_path
                )

            self.embed_dim = 192  # ECAPA-TDNN 輸出 192-dim

            # Projection layer
            if self.output_dim != self.embed_dim:
                self.proj = nn.Linear(self.embed_dim, self.output_dim)
            else:
                self.proj = nn.Identity()

            print(f"✅ Loaded ECAPA-TDNN encoder (output_dim={self.output_dim})")

        except ImportError:
            print("❌ SpeechBrain not installed. Install with: pip install speechbrain")
            raise

    def forward(self, audio):
        """
        Args:
            audio: (B, T) waveform at 24kHz or 16kHz

        Returns:
            embedding: (B, output_dim) speaker embedding
        """
        B = audio.shape[0]
        embeddings = []

        with torch.set_grad_enabled(not self.freeze):
            for i in range(B):
                # 處理單個音頻
                wav = audio[i].cpu().numpy()  # (T,)

                if self.model_type == 'resemblyzer':
                    # Resemblyzer 需要 16kHz
                    # 如果是 24kHz，需要 resample
                    if len(wav) > 48000:  # 假設 > 2秒 @ 24kHz
                        wav = self._resample_to_16k(wav, orig_sr=24000)

                    # Resemblyzer.embed_utterance() 返回 numpy array
                    emb = self.encoder.embed_utterance(wav)  # (256,)
                    emb = torch.from_numpy(emb).float().to(audio.device)

                elif self.model_type == 'ecapa':
                    # ECAPA-TDNN 需要 16kHz
                    if len(wav) > 48000:
                        wav = self._resample_to_16k(wav, orig_sr=24000)

                    # SpeechBrain 需要 tensor 輸入
                    # 注意: ECAPA encoder 預設在 CPU，我們傳入 CPU tensor
                    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)

                    # 提取 embedding (encoder 在 CPU 上處理)
                    emb = self.encoder.encode_batch(wav_tensor)  # (1, 1, 192)
                    emb = emb.squeeze()  # (192,)
                    # 保持在 CPU，稍後統一處理設備轉換

                embeddings.append(emb)

        # Stack to batch
        embeddings = torch.stack(embeddings, dim=0)  # (B, embed_dim)

        # Project to output_dim
        embeddings = self.proj(embeddings)  # (B, output_dim)

        return embeddings

    def _resample_to_16k(self, wav, orig_sr=24000, target_sr=16000):
        """Resample audio from orig_sr to target_sr"""
        import librosa
        wav_resampled = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
        return wav_resampled


# ============================================================================
#                選項 2: 簡單的 CNN-based Speaker Encoder
# ============================================================================

class SimpleSpeakerEncoder(nn.Module):
    """
    簡單的 CNN-based Speaker Encoder

    適用於:
    - 快速實驗
    - 無法安裝 resemblyzer/speechbrain 時

    Note: 泛化能力不如預訓練模型，建議僅用於概念驗證
    """

    def __init__(self, output_dim=256):
        super().__init__()

        # Input: waveform (B, T) -> (B, 1, T)

        # Conv layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=16)  # ~24kHz->1.5kHz
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm1d(512)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # FC layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, output_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, audio):
        """
        Args:
            audio: (B, T) waveform

        Returns:
            embedding: (B, output_dim) speaker embedding
        """
        # (B, T) -> (B, 1, T)
        x = audio.unsqueeze(1)

        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, T')
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, T'')
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 256, T''')
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 512, T'''')

        # Global pooling
        x = self.pool(x)  # (B, 512, 1)
        x = x.squeeze(-1)  # (B, 512)

        # FC layers
        x = F.relu(self.fc1(x))  # (B, 256)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, output_dim)

        return x


# ============================================================================
#                        對比學習損失（可選）
# ============================================================================

class ContrastiveSpeakerLoss(nn.Module):
    """
    對比學習損失，用於訓練 noise-robust speaker encoder

    目標: 同 speaker 的 noisy/clean 應該接近
          不同 speaker 應該遠離

    使用 InfoNCE loss (NT-Xent)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, noisy_emb, clean_emb, speaker_ids):
        """
        Args:
            noisy_emb: (B, D) embeddings from noisy audio
            clean_emb: (B, D) embeddings from clean audio
            speaker_ids: (B,) speaker labels

        Returns:
            loss: scalar contrastive loss
        """
        B = noisy_emb.shape[0]

        # Normalize embeddings
        noisy_emb = F.normalize(noisy_emb, dim=-1)  # (B, D)
        clean_emb = F.normalize(clean_emb, dim=-1)  # (B, D)

        # Compute similarity matrix
        # sim[i, j] = similarity between noisy_emb[i] and clean_emb[j]
        sim_matrix = torch.mm(noisy_emb, clean_emb.t()) / self.temperature  # (B, B)

        # Positive pairs: same speaker (diagonal elements)
        # Negative pairs: different speakers (off-diagonal)

        # Labels: each noisy sample should match its corresponding clean sample
        labels = torch.arange(B, device=noisy_emb.device)

        # InfoNCE loss
        loss = self.criterion(sim_matrix, labels)

        return loss


# ============================================================================
#                            工具函數
# ============================================================================

def create_speaker_encoder(model_type='resemblyzer', freeze=True, output_dim=256):
    """
    工廠函數：創建 speaker encoder

    Args:
        model_type: 'resemblyzer', 'ecapa', 或 'simple'
        freeze: 是否凍結參數
        output_dim: 輸出維度

    Returns:
        speaker_encoder: nn.Module
    """
    if model_type in ['resemblyzer', 'ecapa']:
        return PretrainedSpeakerEncoder(
            model_type=model_type,
            freeze=freeze,
            output_dim=output_dim
        )
    elif model_type == 'simple':
        return SimpleSpeakerEncoder(output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
#                            測試代碼
# ============================================================================

if __name__ == '__main__':
    print("測試 Speaker Encoder...")

    # 模擬音頻輸入
    batch_size = 4
    audio_len = 24000 * 3  # 3 seconds @ 24kHz
    audio = torch.randn(batch_size, audio_len)

    print(f"\nInput audio shape: {audio.shape}")

    # 測試 Simple Encoder
    print("\n" + "="*60)
    print("Testing SimpleSpeakerEncoder...")
    print("="*60)

    simple_encoder = SimpleSpeakerEncoder(output_dim=256)
    simple_emb = simple_encoder(audio)
    print(f"Output embedding shape: {simple_emb.shape}")
    print(f"Embedding stats: mean={simple_emb.mean().item():.4f}, std={simple_emb.std().item():.4f}")

    # 測試 Pretrained Encoder (如果可用)
    print("\n" + "="*60)
    print("Testing PretrainedSpeakerEncoder (resemblyzer)...")
    print("="*60)

    try:
        pretrained_encoder = create_speaker_encoder(
            model_type='resemblyzer',
            freeze=True,
            output_dim=256
        )
        pretrained_emb = pretrained_encoder(audio)
        print(f"Output embedding shape: {pretrained_emb.shape}")
        print(f"Embedding stats: mean={pretrained_emb.mean().item():.4f}, std={pretrained_emb.std().item():.4f}")
    except Exception as e:
        print(f"⚠️ Pretrained encoder not available: {e}")

    # 測試 Contrastive Loss
    print("\n" + "="*60)
    print("Testing ContrastiveSpeakerLoss...")
    print("="*60)

    noisy_emb = torch.randn(batch_size, 256)
    clean_emb = torch.randn(batch_size, 256)
    speaker_ids = torch.tensor([0, 1, 0, 2])  # speaker labels

    contrastive_loss = ContrastiveSpeakerLoss(temperature=0.07)
    loss = contrastive_loss(noisy_emb, clean_emb, speaker_ids)
    print(f"Contrastive loss: {loss.item():.4f}")

    print("\n✅ All tests passed!")

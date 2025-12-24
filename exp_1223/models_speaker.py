"""
exp_1223: Speaker-Conditioned LoRA Model

在 WavTokenizer Encoder 的 LoRA fine-tuning 基礎上，
增加 Speaker Conditioning 機制

設計思路:
1. 保持 LoRA fine-tuning 架構
2. 將 speaker embedding 注入到 encoder features 中
3. 兩種 speaker conditioning 方式:
   - 方式 A: FiLM (Feature-wise Linear Modulation)
   - 方式 B: Cross-Attention (參考 c_code/exp3-1)

本實現選擇 FiLM，因為:
- 更輕量，不增加太多參數
- 與 LoRA 的低秩適應概念一致
- 計算效率更高
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from exp_1217.models import TeacherStudentConfigurableLoRA


class SpeakerFiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for Speaker Conditioning

    FiLM: y = γ * x + β
    其中 γ, β 由 speaker embedding 生成

    參考: https://arxiv.org/abs/1709.07871

    改進 (2024-12-24):
    - 加入 speaker embedding L2 normalization，讓不同 speaker 處於同一尺度
    - 這有助於 FiLM 對 unseen speakers 的泛化
    """

    def __init__(self, speaker_dim: int = 256, feature_dim: int = 512, hidden_dim: int = 256,
                 normalize_speaker: bool = True):
        super().__init__()

        self.speaker_dim = speaker_dim
        self.feature_dim = feature_dim
        self.normalize_speaker = normalize_speaker

        # Speaker embedding → FiLM parameters (γ, β)
        self.film_generator = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim * 2),  # γ and β
        )

        # Initialize to identity transformation
        # γ = 1, β = 0 initially
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        self.film_generator[-1].bias.data[:feature_dim] = 1.0  # γ = 1

    def forward(self, features: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D, T) encoder features (WavTokenizer format)
            speaker_embedding: (B, speaker_dim)

        Returns:
            modulated_features: (B, D, T)
        """
        # L2 normalize speaker embedding for better generalization
        if self.normalize_speaker:
            speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)

        # Generate FiLM parameters
        film_params = self.film_generator(speaker_embedding)  # (B, D*2)
        gamma = film_params[:, :self.feature_dim]  # (B, D)
        beta = film_params[:, self.feature_dim:]   # (B, D)

        # Expand for broadcasting with (B, D, T) format
        gamma = gamma.unsqueeze(2)  # (B, D, 1)
        beta = beta.unsqueeze(2)    # (B, D, 1)

        # Apply FiLM
        modulated = gamma * features + beta

        return modulated


class SpeakerCrossAttention(nn.Module):
    """
    Cross-Attention for Speaker Conditioning
    (參考 c_code/exp3-1 的設計)

    Token features (Query) attend to Speaker embedding (Key, Value)

    改進 (2024-12-24):
    - 加入 speaker embedding L2 normalization
    """

    def __init__(self, feature_dim: int = 512, speaker_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1, normalize_speaker: bool = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.normalize_speaker = normalize_speaker

        # Project speaker embedding to feature space
        self.speaker_proj = nn.Linear(speaker_dim, feature_dim)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)

        # Learnable gate (optional)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D, T) encoder features (WavTokenizer format)
            speaker_embedding: (B, speaker_dim)

        Returns:
            conditioned_features: (B, D, T)
        """
        # L2 normalize speaker embedding for better generalization
        if self.normalize_speaker:
            speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)

        # Transpose to (B, T, D) for attention
        features_t = features.transpose(1, 2)  # (B, T, D)

        # Project speaker to feature space
        speaker_feat = self.speaker_proj(speaker_embedding)  # (B, D)
        speaker_feat = speaker_feat.unsqueeze(1)  # (B, 1, D)

        # Cross-attention: features attend to speaker
        attn_output, _ = self.cross_attention(
            query=features_t,    # (B, T, D)
            key=speaker_feat,    # (B, 1, D)
            value=speaker_feat   # (B, 1, D)
        )  # (B, T, D)

        attn_output = self.norm(attn_output)

        # Learnable gate: α * features + (1-α) * attn_output
        gate_input = torch.cat([features_t, attn_output], dim=-1)  # (B, T, D*2)
        alpha = self.gate(gate_input)  # (B, T, 1)

        conditioned = alpha * features_t + (1 - alpha) * attn_output  # (B, T, D)

        # Transpose back to (B, D, T)
        return conditioned.transpose(1, 2)


class TeacherStudentSpeakerConditioned(nn.Module):
    """
    Speaker-Conditioned Teacher-Student Model

    結合:
    1. WavTokenizer Encoder with LoRA fine-tuning
    2. Speaker conditioning (FiLM or Cross-Attention)
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.2,
        lora_layers: str = 'all_18',
        speaker_condition_type: str = 'film',  # 'film' or 'cross_attention'
        speaker_dim: int = 256,
        feature_dim: int = 512,
        device: str = 'cuda',
    ):
        super().__init__()

        self.speaker_condition_type = speaker_condition_type

        # 載入基礎 Teacher-Student 模型 (with LoRA)
        self.base_model = TeacherStudentConfigurableLoRA(
            wavtok_config=wavtok_config,
            wavtok_ckpt=wavtok_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_layers=lora_layers,
            device=device,
        )

        # Speaker conditioning module
        if speaker_condition_type == 'film':
            self.speaker_condition = SpeakerFiLM(
                speaker_dim=speaker_dim,
                feature_dim=feature_dim,
            )
        elif speaker_condition_type == 'cross_attention':
            self.speaker_condition = SpeakerCrossAttention(
                feature_dim=feature_dim,
                speaker_dim=speaker_dim,
            )
        else:
            raise ValueError(f"Unknown speaker_condition_type: {speaker_condition_type}")

        self.speaker_condition = self.speaker_condition.to(device)

        # Copy references for compatibility
        self.teacher = self.base_model.teacher
        self.student = self.base_model.student

        print(f"Speaker conditioning: {speaker_condition_type}")
        speaker_params = sum(p.numel() for p in self.speaker_condition.parameters())
        print(f"  Speaker condition params: {speaker_params:,}")

    def forward(self, noisy_audio: torch.Tensor, clean_audio: torch.Tensor,
                speaker_embedding: torch.Tensor = None):
        """
        Args:
            noisy_audio: (B, T_audio)
            clean_audio: (B, T_audio)
            speaker_embedding: (B, speaker_dim) - 如果為 None 則不使用 speaker conditioning

        Returns:
            dict with:
                - student_encoder_out: speaker-conditioned student features
                - teacher_encoder_out: teacher features (no conditioning)
                - student_codes: VQ codes from conditioned features
                - teacher_codes: VQ codes from teacher
                - codebook: VQ codebook
        """
        # 1. 基礎 forward (without speaker conditioning)
        base_output = self.base_model(noisy_audio, clean_audio)

        # 2. 如果提供了 speaker embedding，對 student features 進行 conditioning
        if speaker_embedding is not None:
            student_features = base_output['student_encoder_out']

            # Apply speaker conditioning
            conditioned_features = self.speaker_condition(student_features, speaker_embedding)

            # 重新計算 VQ codes (使用 conditioned features)
            # 使用與 base_model 相同的 quantizer 調用方式
            with torch.no_grad():
                quantizer = self.base_model.student.feature_extractor.encodec.quantizer
                # quantizer.forward 參數: (x, frame_rate, bandwidth, ...)
                q_res = quantizer.forward(conditioned_features, frame_rate=75, bandwidth=0.75)
                conditioned_codes = q_res.codes  # (n_q, B, T)

            # 更新輸出
            base_output['student_encoder_out'] = conditioned_features
            base_output['student_codes'] = conditioned_codes

        return base_output

    def train(self, mode=True):
        """Override train to keep teacher frozen"""
        super().train(mode)
        self.base_model.teacher.eval()
        self.base_model.teacher.feature_extractor.encodec.quantizer.eval()
        self.base_model.student.feature_extractor.encodec.quantizer.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Only save trainable parameters (LoRA + speaker condition)"""
        full_state = super().state_dict(*args, **kwargs)

        trainable_keys = [
            k for k, v in self.named_parameters()
            if v.requires_grad
        ]

        return {k: v for k, v in full_state.items() if k in trainable_keys}

    def load_state_dict(self, state_dict, strict=False):
        """Load only trainable parameters"""
        return super().load_state_dict(state_dict, strict=False)

"""
exp_0112_adapter: Exp J - 中層 Adapter 去噪實驗

核心設計:
- 在噪音最敏感的 L5-L8 中層插入可訓練的 Adapter 模組
- Adapter 專門學習去噪，不影響原始 WavTokenizer 權重
- 保持全層梯度流通，但去噪能力集中在 Adapter

Adapter 架構:
┌─────────────────────────────────────────────────────────────┐
│  Input (from L4 downsample)                                 │
│     ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  MidLayerAdapter                                     │   │
│  │  ├─ LayerNorm                                        │   │
│  │  ├─ Conv1d (down_proj: C → C//4)                    │   │
│  │  ├─ GELU                                             │   │
│  │  ├─ Conv1d (up_proj: C//4 → C)                      │   │
│  │  └─ Residual: output = input + scale * adapter_out   │   │
│  └─────────────────────────────────────────────────────┘   │
│     ↓                                                       │
│  L5-L8 (原始層，權重凍結或輕量 LoRA)                       │
│     ↓                                                       │
│  Output (to L9)                                             │
└─────────────────────────────────────────────────────────────┘

為什麼這樣設計:
1. 位置: L5-L8 是噪音破壞最嚴重的區域 (cos_sim = 0.21-0.29)
2. Bottleneck: 降維後再升維，限制容量避免過擬合
3. Residual: 保持原始信息流，Adapter 只做增量修正
4. 可學習 scale: 控制 Adapter 的影響程度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from decoder.pretrained import WavTokenizer


class CodebookDriftError(Exception):
    """Codebook 漂移錯誤"""
    pass


# ============================================================
# Adapter 模組定義
# ============================================================

class DenoiseAdapter(nn.Module):
    """
    去噪 Adapter 模組

    設計:
    - Bottleneck 結構: input_dim → hidden_dim → input_dim
    - 使用 Conv1d 處理時序特徵
    - 可學習的 scale 參數控制影響程度
    - Residual 連接保持原始信息
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        reduction_factor: int = 4,
        dropout: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim // reduction_factor

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Adapter 結構
        self.layer_norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

        # 可學習的 scale 參數 (初始值小，讓訓練初期 Adapter 影響小)
        self.scale = nn.Parameter(torch.tensor(init_scale))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """使用小值初始化，確保訓練初期 Adapter 輸出接近零"""
        nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)  # 零初始化，輸出接近零
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) encoder 中間層輸出

        Returns:
            (B, C, T) 去噪後的特徵
        """
        # 保存殘差
        residual = x

        # LayerNorm (需要轉換維度)
        # (B, C, T) -> (B, T, C) -> LayerNorm -> (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)

        # Bottleneck
        x = self.down_proj(x)      # (B, C, T) -> (B, H, T)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)        # (B, H, T) -> (B, C, T)

        # Residual 連接 + 可學習 scale
        output = residual + self.scale * x

        return output


class MultiHeadDenoiseAdapter(nn.Module):
    """
    多頭去噪 Adapter (可選的更強版本)

    類似 Multi-Head Attention 的設計，但用於去噪
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        hidden_dim: int = None,
        dropout: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim // 4

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.layer_norm = nn.LayerNorm(input_dim)

        # 多頭投影
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, self.head_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(self.head_dim, input_dim // num_heads, kernel_size=1),
            )
            for _ in range(num_heads)
        ])

        # 輸出投影
        self.output_proj = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(init_scale))

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # LayerNorm
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)

        # 多頭處理
        head_outputs = [head(x) for head in self.heads]
        x = torch.cat(head_outputs, dim=1)  # (B, C, T)

        x = self.output_proj(x)

        return residual + self.scale * x


# ============================================================
# Encoder with Adapter 包裝器
# ============================================================

class EncoderWithAdapter(nn.Module):
    """
    在原始 Encoder 中插入 Adapter 的包裝器

    插入位置: L4 (downsample) 之後，L5 之前
    這樣 Adapter 可以在噪音敏感層 (L5-L8) 之前先處理特徵
    """

    def __init__(
        self,
        original_encoder: nn.Module,
        adapter_position: int = 4,  # 在 model[4] 之前插入
        adapter_dim: int = 128,     # L4 輸出的 channel 數
        adapter_type: str = 'simple',  # 'simple' or 'multihead'
        adapter_hidden_dim: int = None,
        adapter_dropout: float = 0.1,
        adapter_init_scale: float = 0.01,
    ):
        super().__init__()

        self.original_encoder = original_encoder
        self.adapter_position = adapter_position

        # 創建 Adapter
        if adapter_type == 'simple':
            self.adapter = DenoiseAdapter(
                input_dim=adapter_dim,
                hidden_dim=adapter_hidden_dim,
                dropout=adapter_dropout,
                init_scale=adapter_init_scale,
            )
        elif adapter_type == 'multihead':
            self.adapter = MultiHeadDenoiseAdapter(
                input_dim=adapter_dim,
                hidden_dim=adapter_hidden_dim,
                dropout=adapter_dropout,
                init_scale=adapter_init_scale,
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        print(f"Adapter inserted before layer {adapter_position}")
        print(f"  Type: {adapter_type}")
        print(f"  Input dim: {adapter_dim}")
        print(f"  Hidden dim: {adapter_hidden_dim or adapter_dim // 4}")
        print(f"  Trainable params: {sum(p.numel() for p in self.adapter.parameters()):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adapter injection

        Args:
            x: (B, 1, T) 輸入音頻

        Returns:
            (B, C, T') encoder 輸出
        """
        # 逐層前向傳播，在指定位置插入 Adapter
        for i, layer in enumerate(self.original_encoder.model):
            x = layer(x)

            # 在指定位置後插入 Adapter
            if i == self.adapter_position - 1:
                x = self.adapter(x)

        return x


# ============================================================
# Teacher-Student with Adapter 模型
# ============================================================

class TeacherStudentAdapter(nn.Module):
    """
    Exp J: 中層 Adapter 去噪模型

    設計:
    - Teacher: 原始 WavTokenizer (完全凍結)
    - Student: WavTokenizer + Adapter (只訓練 Adapter)
    - Adapter 插入在 L4-L5 之間，專門處理噪音敏感區域的輸入
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        adapter_position: int = 4,
        adapter_dim: int = 128,
        adapter_type: str = 'simple',
        adapter_hidden_dim: int = None,
        adapter_dropout: float = 0.1,
        adapter_init_scale: float = 0.01,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Teacher: 完全凍結
        print("=" * 60)
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)
        self._freeze_quantizer(self.teacher, "Teacher")

        # Student: 原始權重凍結 + Adapter 可訓練
        print("=" * 60)
        print("Loading Student with Adapter...")
        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        # 凍結 Student 的所有原始權重
        for param in self.student.parameters():
            param.requires_grad = False

        # 獲取 L4 輸出的 channel 數
        # L4 是 model[3] (第4個，0-indexed)，它是 downsample 層
        # 我們需要查看 L5 的輸入 channel
        l5_layer = self.student.feature_extractor.encodec.encoder.model[4]
        if hasattr(l5_layer, 'block'):
            adapter_dim = l5_layer.block[1].conv.conv.in_channels
        else:
            adapter_dim = l5_layer.conv.conv.in_channels

        print(f"  Detected adapter_dim from L5 input: {adapter_dim}")

        # 替換 encoder 為帶 Adapter 的版本
        original_encoder = self.student.feature_extractor.encodec.encoder
        self.student.feature_extractor.encodec.encoder = EncoderWithAdapter(
            original_encoder=original_encoder,
            adapter_position=adapter_position,
            adapter_dim=adapter_dim,
            adapter_type=adapter_type,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout=adapter_dropout,
            adapter_init_scale=adapter_init_scale,
        )

        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # 獲取 codebook 並保存初始狀態
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        # 統計可訓練參數
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.student.parameters())

        print("=" * 60)
        print(f"Codebook shape: {self.codebook.shape}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
        print("=" * 60)

    def _freeze_quantizer(self, model, name: str):
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen")

    def get_adapter(self) -> nn.Module:
        """獲取 Adapter 模組"""
        return self.student.feature_extractor.encodec.encoder.adapter

    def get_adapter_params(self) -> List[nn.Parameter]:
        """獲取 Adapter 參數 (用於 optimizer)"""
        return list(self.get_adapter().parameters())

    def _get_codebook(self) -> torch.Tensor:
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        codebook = quantizer.vq.layers[0].codebook.detach().clone()
        return codebook

    def _get_teacher_codebook(self) -> torch.Tensor:
        return self.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def _get_student_codebook(self) -> torch.Tensor:
        return self.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def train(self, mode: bool = True):
        super().train(mode)
        # Teacher 始終 eval
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()
        # Student 的 quantizer 也 eval
        self.student.feature_extractor.encodec.quantizer.eval()
        return self

    def check_codebook_integrity(self, raise_error: bool = True) -> dict:
        teacher_cb = self._get_teacher_codebook()
        student_cb = self._get_student_codebook()

        teacher_drift = (self._initial_teacher_codebook - teacher_cb).abs().mean().item()
        student_drift = (self._initial_student_codebook - student_cb).abs().mean().item()

        result = {
            'teacher_drift': teacher_drift,
            'student_drift': student_drift,
            'teacher_ok': teacher_drift < 1e-7,
            'student_ok': student_drift < 1e-7,
        }

        if raise_error:
            if teacher_drift > 1e-7:
                raise CodebookDriftError(f"Teacher codebook drift: {teacher_drift:.8f}")
            if student_drift > 1e-7:
                raise CodebookDriftError(f"Student codebook drift: {student_drift:.8f}")

        return result

    def forward(self, noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> dict:
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward (clean audio)
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward (noisy audio, with Adapter)
        student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)

        self.student.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            quantizer = self.student.feature_extractor.encodec.quantizer
            student_vq = quantizer(student_encoder_out, frame_rate=75, bandwidth=0.075)
            student_codes = student_vq.codes

        return {
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
            'adapter_scale': self.get_adapter().scale.item(),
        }

    def compute_ce_logits(self, encoder_out):
        B, C, T = encoder_out.shape
        z = encoder_out.permute(0, 2, 1)
        logits = 2 * torch.matmul(z, self.codebook.t())
        c_sq = (self.codebook ** 2).sum(dim=1)
        logits = logits - c_sq.unsqueeze(0).unsqueeze(0)
        logits = logits.permute(0, 2, 1)
        return logits

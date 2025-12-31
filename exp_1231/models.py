"""
exp_1231: 優化方向測試 - Models

包含：
- DenoisingAdapter: Exp73 專用的去噪 Adapter
- StudentEncoderWithAdapter: 整合 Adapter 的 Student Encoder
- MultiScaleEncoder: 支援提取中間層特徵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class DenoisingAdapter(nn.Module):
    """
    Exp73: Denoising Adapter

    在 encoder 輸出後加入專門的去噪模組。
    學習 noisy → clean 的殘差映射。
    """

    def __init__(
        self,
        dim: int = 512,
        expansion: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.dim = dim
        hidden = dim * expansion

        # 多層 adapter
        layers = []
        for i in range(num_layers):
            layers.append(AdapterBlock(dim, hidden, dropout))
        self.layers = nn.ModuleList(layers)

        # 可學習的殘差權重
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) encoder output

        Returns:
            denoised: (B, T, D)
        """
        residual = x
        for layer in self.layers:
            x = layer(x)

        # 加權殘差連接
        return residual + self.alpha * x


class AdapterBlock(nn.Module):
    """單個 Adapter Block"""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x


class ConvDenoisingAdapter(nn.Module):
    """
    卷積版本的 Denoising Adapter

    使用 1D 卷積處理時序依賴，可能更適合音訊。
    """

    def __init__(
        self,
        dim: int = 512,
        kernel_size: int = 5,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        layers = []
        for i in range(num_layers):
            layers.append(
                ConvAdapterBlock(dim, kernel_size, dropout)
            )
        self.layers = nn.ModuleList(layers)

        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        residual = x
        # 轉換為 (B, D, T) for conv1d
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)

        # 轉回 (B, T, D)
        x = x.transpose(1, 2)
        return residual + self.alpha * x


class ConvAdapterBlock(nn.Module):
    """卷積 Adapter Block"""

    def __init__(self, dim: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2

        self.norm = nn.GroupNorm(1, dim)  # LayerNorm for 1D
        self.conv1 = nn.Conv1d(dim, dim * 2, kernel_size, padding=padding)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(dim * 2, dim, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T)"""
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return residual + x


class TeacherStudentWithAdapter(nn.Module):
    """
    Exp73: 整合 Adapter 的 Teacher-Student 模型

    基於 TeacherStudentConfigurableLoRA，加入 DenoisingAdapter。
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        lora_dropout: float = 0.2,
        lora_layers: str = 'all_18',
        adapter_type: str = 'mlp',  # 'mlp' or 'conv'
        adapter_dim: int = 512,
        adapter_expansion: int = 4,
        adapter_num_layers: int = 2,
        adapter_dropout: float = 0.1,
        device: str = 'cuda',
    ):
        super().__init__()

        # 載入基礎模型
        from exp_1217.models import TeacherStudentConfigurableLoRA
        self.base_model = TeacherStudentConfigurableLoRA(
            wavtok_config=wavtok_config,
            wavtok_ckpt=wavtok_ckpt,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_layers=lora_layers,
            device=device,
        )

        # 加入 Adapter
        if adapter_type == 'mlp':
            self.adapter = DenoisingAdapter(
                dim=adapter_dim,
                expansion=adapter_expansion,
                dropout=adapter_dropout,
                num_layers=adapter_num_layers,
            )
        elif adapter_type == 'conv':
            self.adapter = ConvDenoisingAdapter(
                dim=adapter_dim,
                kernel_size=5,
                num_layers=adapter_num_layers,
                dropout=adapter_dropout,
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        self.adapter = self.adapter.to(device)

        # 代理屬性
        self.teacher = self.base_model.teacher
        self.student = self.base_model.student

    def forward(
        self,
        noisy_audio: torch.Tensor,
        clean_audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adapter

        Returns:
            Dict with student/teacher features and codes
        """
        # 基礎模型 forward
        output = self.base_model(noisy_audio, clean_audio)

        # 對 student features 應用 adapter
        student_features_adapted = self.adapter(output['student_encoder_out'])

        # 用 adapted features 重新計算 codes
        student_codes_adapted = self._get_codes(
            student_features_adapted,
            output['codebook']
        )

        # 更新 output
        output['student_encoder_out_raw'] = output['student_encoder_out']
        output['student_encoder_out'] = student_features_adapted
        output['student_codes_raw'] = output['student_codes']
        output['student_codes'] = student_codes_adapted

        return output

    def _get_codes(self, features: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """計算最近的 codebook indices"""
        B, T, D = features.shape
        flat = features.reshape(-1, D)
        distances = torch.cdist(flat, codebook, p=2)
        codes = distances.argmin(dim=-1)
        return codes.reshape(B, T)

    def parameters(self, recurse=True):
        """只返回可訓練的參數"""
        # LoRA 參數
        for param in self.base_model.parameters():
            if param.requires_grad:
                yield param
        # Adapter 參數
        for param in self.adapter.parameters():
            yield param

    def named_parameters(self, prefix='', recurse=True):
        """返回命名的可訓練參數"""
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                yield name, param
        for name, param in self.adapter.named_parameters(prefix='adapter'):
            yield name, param

    def state_dict(self, *args, **kwargs):
        """保存模型狀態"""
        return {
            'base_model': self.base_model.state_dict(*args, **kwargs),
            'adapter': self.adapter.state_dict(*args, **kwargs),
        }

    def load_state_dict(self, state_dict, strict=True):
        """載入模型狀態"""
        self.base_model.load_state_dict(state_dict['base_model'], strict=strict)
        self.adapter.load_state_dict(state_dict['adapter'], strict=strict)


def extract_intermediate_features(
    encoder,
    audio: torch.Tensor,
    layer_indices: List[int] = [6, 12, 18],
    bandwidth_id: int = 0,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Exp74: 提取中間層特徵

    Args:
        encoder: WavTokenizer encoder
        audio: (B, L) or (B, 1, L)
        layer_indices: 要提取的層索引
        bandwidth_id: bandwidth ID

    Returns:
        intermediate_features: List of (B, T, D) tensors
        final_features: (B, T, D)
    """
    # 這個函數需要根據實際的 WavTokenizer 結構來實作
    # 這裡提供一個框架

    # 獲取 encoder 的內部層
    # 注意：需要根據實際的 WavTokenizer 結構調整
    raise NotImplementedError(
        "需要根據 WavTokenizer 的具體結構實作。"
        "可能需要修改 WavTokenizer 來支援中間層提取。"
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """計算模型參數數量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

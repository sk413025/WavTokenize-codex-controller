"""
exp_0113: 增強版 Adapter 去噪實驗

包含兩種模型:
1. Exp L: 多位置 Adapter (TeacherStudentMultiAdapter)
2. Exp M: Adapter + LoRA 混合 (TeacherStudentAdapterLoRA)

設計理念:
- Exp L: 增加 Adapter 數量和容量，覆蓋更多噪音敏感區域
- Exp M: 結合 Adapter 的聽感優勢和 LoRA 的數值表現
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

    Args:
        input_dim: 輸入維度
        hidden_dim: Bottleneck 中間維度 (預設 input_dim // reduction_factor)
        reduction_factor: 降維倍數 (預設 2，比 Exp J 的 4 更大)
        dropout: Dropout 比例
        init_scale: 初始 scale 值
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        reduction_factor: int = 2,  # 比 Exp J 的 4 更大
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

        # 可學習的 scale 參數
        self.scale = nn.Parameter(torch.tensor(init_scale))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """使用小值初始化，確保訓練初期 Adapter 輸出接近零"""
        nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) encoder 中間層輸出

        Returns:
            (B, C, T) 去噪後的特徵
        """
        residual = x

        # LayerNorm (需要轉換維度)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)

        # Bottleneck
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        # Residual 連接 + 可學習 scale
        return residual + self.scale * x


# ============================================================
# LoRA 模組定義
# ============================================================

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 層

    將原始 Conv1d 層轉換為 LoRA 可訓練版本

    Args:
        original_layer: 原始 Conv1d 層
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout
    """

    def __init__(
        self,
        original_layer: nn.Conv1d,
        rank: int = 32,
        alpha: float = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_channels = original_layer.in_channels
        out_channels = original_layer.out_channels
        kernel_size = original_layer.kernel_size[0]

        # LoRA 分解: W = W0 + BA
        # A: (in_channels * kernel_size) -> rank
        # B: rank -> (out_channels * kernel_size)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_channels * kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels * kernel_size, rank))

        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 凍結原始層
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始輸出 (凍結)
        original_out = self.original_layer(x)

        # LoRA 輸出
        # x: (B, C_in, T) -> unfold -> (B, C_in * K, T') -> matmul
        B, C_in, T = x.shape
        K = self.original_layer.kernel_size[0]
        padding = self.original_layer.padding[0]
        stride = self.original_layer.stride[0]

        # 使用 unfold 提取 patches
        x_unfold = F.unfold(
            x.unsqueeze(2),  # (B, C_in, 1, T)
            kernel_size=(1, K),
            padding=(0, padding),
            stride=(1, stride)
        )  # (B, C_in * K, T')

        T_out = x_unfold.shape[2]

        # LoRA: x_unfold @ A^T @ B^T
        x_dropout = self.dropout(x_unfold)
        lora_out = torch.matmul(x_dropout.permute(0, 2, 1), self.lora_A.T)  # (B, T', rank)
        lora_out = torch.matmul(lora_out, self.lora_B.T)  # (B, T', C_out * K)

        # 重塑為 Conv 輸出形狀
        C_out = self.original_layer.out_channels
        lora_out = lora_out.permute(0, 2, 1)  # (B, C_out * K, T')

        # 簡化：假設 kernel_size=1 或處理較複雜的情況
        if K == 1:
            lora_out = lora_out.view(B, C_out, T_out)
        else:
            # 對於 kernel_size > 1，取平均
            lora_out = lora_out.view(B, C_out, K, T_out).mean(dim=2)

        return original_out + self.scaling * lora_out


class LoRAConv1d(nn.Module):
    """
    簡化版 LoRA Conv1d

    直接使用兩個小 Conv1d 代替矩陣分解

    Args:
        original_conv: 原始 Conv1d 層
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
    """

    def __init__(
        self,
        original_conv: nn.Conv1d,
        rank: int = 32,
        alpha: float = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_ch = original_conv.in_channels
        out_ch = original_conv.out_channels
        kernel_size = original_conv.kernel_size[0]
        padding = original_conv.padding[0]

        # LoRA 分解: 兩個小 Conv1d
        self.lora_down = nn.Conv1d(in_ch, rank, kernel_size=kernel_size, padding=padding, bias=False)
        self.lora_up = nn.Conv1d(rank, out_ch, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # 凍結原始層
        for param in self.original_conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_conv(x)

        lora_out = self.lora_down(x)
        lora_out = self.dropout(lora_out)
        lora_out = self.lora_up(lora_out)

        return original_out + self.scaling * lora_out


# ============================================================
# Exp L: 多位置 Adapter 模型
# ============================================================

class EncoderWithMultiAdapter(nn.Module):
    """
    在多個位置插入 Adapter 的 Encoder 包裝器

    WavTokenizer Encoder 層結構 (18層):
    - L0: SConv1d (1→32)
    - L1: SEANetResnetBlock (32)
    - L2: ELU
    - L3: SConv1d (32→64, stride=2)
    - L4: SEANetResnetBlock (64)
    - L5: ELU
    - L6: SConv1d (64→128, stride=4)
    - L7: SEANetResnetBlock (128)
    - L8: ELU
    - L9: SConv1d (128→256, stride=5)
    - L10: SEANetResnetBlock (256)
    - ...

    建議 Adapter 位置: 在 ResBlock 後 (L1, L4, L7, L10...)

    Args:
        original_encoder: 原始 WavTokenizer encoder
        adapter_positions: Adapter 插入位置列表 (在這些層之後插入)
        adapter_dims: 每個位置對應的 channel 數 (可選，會自動推斷)
        reduction_factor: Bottleneck 降維倍數
        dropout: Adapter dropout
        init_scale: 初始 scale
    """

    # WavTokenizer Encoder 各層輸出維度對照表
    LAYER_DIMS = {
        0: 32, 1: 32, 2: 32,      # 第一段: 32 channels
        3: 64, 4: 64, 5: 64,      # 第二段: 64 channels
        6: 128, 7: 128, 8: 128,   # 第三段: 128 channels
        9: 256, 10: 256, 11: 256, # 第四段: 256 channels
        12: 512, 13: 512, 14: 512, 15: 512, 16: 512, 17: 512,  # 第五段: 512 channels
    }

    def __init__(
        self,
        original_encoder: nn.Module,
        adapter_positions: List[int] = [1, 4, 7, 10],  # 改為 ResBlock 後的位置
        adapter_dims: Dict[int, int] = None,
        reduction_factor: int = 2,
        dropout: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()

        self.original_encoder = original_encoder
        self.adapter_positions = adapter_positions

        # 創建 Adapters
        self.adapters = nn.ModuleDict()

        for pos in adapter_positions:
            # 獲取該位置的 channel 數
            if adapter_dims and pos in adapter_dims:
                dim = adapter_dims[pos]
            else:
                # 使用預定義的維度表
                dim = self.LAYER_DIMS.get(pos, 128)

            self.adapters[str(pos)] = DenoiseAdapter(
                input_dim=dim,
                reduction_factor=reduction_factor,
                dropout=dropout,
                init_scale=init_scale,
            )

            print(f"  Adapter @ L{pos}: {dim} → {dim // reduction_factor} → {dim}")

        total_adapter_params = sum(
            sum(p.numel() for p in adapter.parameters())
            for adapter in self.adapters.values()
        )
        print(f"  Total Adapter params: {total_adapter_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播，在指定位置插入 Adapter

        Args:
            x: (B, 1, T) 輸入音頻

        Returns:
            (B, C, T') encoder 輸出
        """
        for i, layer in enumerate(self.original_encoder.model):
            x = layer(x)

            # 在指定位置後插入 Adapter
            if i in self.adapter_positions:
                x = self.adapters[str(i)](x)

        return x

    def get_adapter_scales(self) -> Dict[str, float]:
        """獲取所有 Adapter 的 scale 值"""
        return {
            f"L{pos}": self.adapters[str(pos)].scale.item()
            for pos in self.adapter_positions
        }


class TeacherStudentMultiAdapter(nn.Module):
    """
    Exp L: 多位置 Adapter 去噪模型

    設計:
    - Teacher: 原始 WavTokenizer (完全凍結)
    - Student: WavTokenizer + 多個 Adapter (只訓練 Adapter)
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        adapter_positions: List[int] = [2, 4, 6, 8],
        reduction_factor: int = 2,
        dropout: float = 0.1,
        init_scale: float = 0.01,
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
        print("Loading Student with Multi-Position Adapters...")
        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        # 凍結 Student 的所有原始權重
        for param in self.student.parameters():
            param.requires_grad = False

        # 替換 encoder 為帶多 Adapter 的版本
        original_encoder = self.student.feature_extractor.encodec.encoder
        self.student.feature_extractor.encodec.encoder = EncoderWithMultiAdapter(
            original_encoder=original_encoder,
            adapter_positions=adapter_positions,
            reduction_factor=reduction_factor,
            dropout=dropout,
            init_scale=init_scale,
        )

        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # 獲取 codebook 並保存初始狀態
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        # 統計參數
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.student.parameters())

        print("=" * 60)
        print(f"Adapter positions: {adapter_positions}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
        print("=" * 60)

    def _freeze_quantizer(self, model, name: str):
        """凍結量化器"""
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen")

    def get_adapters(self) -> nn.ModuleDict:
        """獲取所有 Adapter"""
        return self.student.feature_extractor.encodec.encoder.adapters

    def get_adapter_params(self) -> List[nn.Parameter]:
        """獲取所有 Adapter 參數"""
        params = []
        for adapter in self.get_adapters().values():
            params.extend(list(adapter.parameters()))
        return params

    def get_adapter_scales(self) -> Dict[str, float]:
        """獲取所有 Adapter 的 scale"""
        return self.student.feature_extractor.encodec.encoder.get_adapter_scales()

    def _get_codebook(self) -> torch.Tensor:
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        return quantizer.vq.layers[0].codebook.detach().clone()

    def _get_teacher_codebook(self) -> torch.Tensor:
        return self.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def _get_student_codebook(self) -> torch.Tensor:
        return self.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()
        self.student.feature_extractor.encodec.quantizer.eval()
        return self

    def check_codebook_integrity(self, raise_error: bool = True) -> dict:
        """檢查 codebook 是否漂移"""
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
        """
        前向傳播

        Args:
            noisy_audio: 含噪音音頻
            clean_audio: 乾淨音頻

        Returns:
            dict with encoder outputs, codes, etc.
        """
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

        # Student forward (noisy audio, with Adapters)
        student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)

        self.student.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            student_vq = self.student.feature_extractor.encodec.quantizer(
                student_encoder_out, frame_rate=75, bandwidth=0.075
            )
            student_codes = student_vq.codes

        return {
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
            'adapter_scales': self.get_adapter_scales(),
        }


# ============================================================
# Exp M: Adapter + LoRA 混合模型
# ============================================================

class EncoderWithAdapterLoRA(nn.Module):
    """
    Adapter + LoRA 混合的 Encoder 包裝器

    架構:
    - 淺層 (L1, L4): 大 Adapter (ResBlock 後)
    - 中層 (L7): 小 Adapter (ResBlock 後)
    - 深層 (L10, L13, L16): 微量 LoRA (ResBlock)

    WavTokenizer Encoder 各層輸出維度:
    - L0-L2: 32 channels
    - L3-L5: 64 channels
    - L6-L8: 128 channels
    - L9-L11: 256 channels
    - L12-L17: 512 channels

    Args:
        original_encoder: 原始 encoder
        shallow_adapter_positions: 淺層 Adapter 位置 (建議 ResBlock 後: 1, 4)
        mid_adapter_positions: 中層 Adapter 位置 (建議: 7)
        mid_lora_layers: 中層 LoRA 層 (建議 ResBlock: 4, 7)
        deep_lora_layers: 深層 LoRA 層 (建議 ResBlock: 10, 13, 16)
        mid_lora_rank: 中層 LoRA rank
        deep_lora_rank: 深層 LoRA rank
    """

    # WavTokenizer Encoder 各層輸出維度對照表
    LAYER_DIMS = {
        0: 32, 1: 32, 2: 32,
        3: 64, 4: 64, 5: 64,
        6: 128, 7: 128, 8: 128,
        9: 256, 10: 256, 11: 256,
        12: 512, 13: 512, 14: 512, 15: 512, 16: 512, 17: 512,
    }

    def __init__(
        self,
        original_encoder: nn.Module,
        shallow_adapter_positions: List[int] = [1, 4],  # ResBlock 後
        mid_adapter_positions: List[int] = [7],         # ResBlock 後
        mid_lora_layers: List[int] = [4, 7],            # ResBlock
        deep_lora_layers: List[int] = [10, 13, 16],     # ResBlock
        shallow_reduction: int = 2,
        mid_reduction: int = 4,
        mid_lora_rank: int = 32,
        mid_lora_alpha: int = 64,
        deep_lora_rank: int = 16,
        deep_lora_alpha: int = 32,
        dropout: float = 0.1,
        init_scale: float = 0.01,
    ):
        super().__init__()

        self.original_encoder = original_encoder
        self.shallow_adapter_positions = shallow_adapter_positions
        self.mid_adapter_positions = mid_adapter_positions
        self.mid_lora_layers = mid_lora_layers
        self.deep_lora_layers = deep_lora_layers

        # 創建淺層 Adapter
        self.shallow_adapters = nn.ModuleDict()
        for pos in shallow_adapter_positions:
            dim = self.LAYER_DIMS.get(pos, 128)
            self.shallow_adapters[str(pos)] = DenoiseAdapter(
                input_dim=dim,
                reduction_factor=shallow_reduction,
                dropout=dropout,
                init_scale=init_scale,
            )
            print(f"  Shallow Adapter @ L{pos}: {dim} → {dim // shallow_reduction} → {dim}")

        # 創建中層 Adapter
        self.mid_adapters = nn.ModuleDict()
        for pos in mid_adapter_positions:
            dim = self.LAYER_DIMS.get(pos, 128)
            self.mid_adapters[str(pos)] = DenoiseAdapter(
                input_dim=dim,
                reduction_factor=mid_reduction,
                dropout=dropout,
                init_scale=init_scale,
            )
            print(f"  Mid Adapter @ L{pos}: {dim} → {dim // mid_reduction} → {dim}")

        # 應用 LoRA 到中層
        self.mid_lora_modules = nn.ModuleDict()
        for layer_idx in mid_lora_layers:
            self._apply_lora_to_layer(layer_idx, mid_lora_rank, mid_lora_alpha, dropout, "mid")

        # 應用 LoRA 到深層
        self.deep_lora_modules = nn.ModuleDict()
        for layer_idx in deep_lora_layers:
            if layer_idx < len(self.original_encoder.model):
                self._apply_lora_to_layer(layer_idx, deep_lora_rank, deep_lora_alpha, dropout, "deep")

        # 統計參數
        adapter_params = sum(
            sum(p.numel() for p in adapter.parameters())
            for adapter in list(self.shallow_adapters.values()) + list(self.mid_adapters.values())
        )
        lora_params = sum(
            sum(p.numel() for p in module.parameters() if p.requires_grad)
            for module in list(self.mid_lora_modules.values()) + list(self.deep_lora_modules.values())
        )
        print(f"  Adapter params: {adapter_params:,}")
        print(f"  LoRA params: {lora_params:,}")
        print(f"  Total trainable: {adapter_params + lora_params:,}")

    def _apply_lora_to_layer(self, layer_idx: int, rank: int, alpha: int, dropout: float, zone: str):
        """對指定層應用 LoRA"""
        if layer_idx >= len(self.original_encoder.model):
            return

        layer = self.original_encoder.model[layer_idx]
        lora_dict = self.mid_lora_modules if zone == "mid" else self.deep_lora_modules

        # 找到 Conv1d 層並替換為 LoRA 版本
        if hasattr(layer, 'block'):
            # ResBlock
            for i, sublayer in enumerate(layer.block):
                if hasattr(sublayer, 'conv') and hasattr(sublayer.conv, 'conv'):
                    if isinstance(sublayer.conv.conv, nn.Conv1d):
                        key = f"{layer_idx}_{i}"
                        lora_dict[key] = LoRAConv1d(
                            sublayer.conv.conv,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout
                        )
                        print(f"    LoRA ({zone}) @ L{layer_idx}.block[{i}]: rank={rank}")
        elif hasattr(layer, 'conv'):
            if hasattr(layer.conv, 'conv') and isinstance(layer.conv.conv, nn.Conv1d):
                key = f"{layer_idx}_conv"
                lora_dict[key] = LoRAConv1d(
                    layer.conv.conv,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                print(f"    LoRA ({zone}) @ L{layer_idx}.conv: rank={rank}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        for i, layer in enumerate(self.original_encoder.model):
            # 原始層
            x = layer(x)

            # 淺層 Adapter
            if i in self.shallow_adapter_positions:
                x = self.shallow_adapters[str(i)](x)

            # 中層 Adapter
            if i in self.mid_adapter_positions:
                x = self.mid_adapters[str(i)](x)

            # 注意: LoRA 已經通過替換 Conv1d 層自動生效

        return x

    def get_adapter_scales(self) -> Dict[str, float]:
        """獲取所有 Adapter 的 scale"""
        scales = {}
        for pos, adapter in self.shallow_adapters.items():
            scales[f"shallow_L{pos}"] = adapter.scale.item()
        for pos, adapter in self.mid_adapters.items():
            scales[f"mid_L{pos}"] = adapter.scale.item()
        return scales


class TeacherStudentAdapterLoRA(nn.Module):
    """
    Exp M: Adapter + LoRA 混合去噪模型

    設計:
    - 淺層 (L0-L4): 大 Adapter → 激進去噪
    - 中層 (L5-L8): 小 Adapter + 小 LoRA → 細化去噪
    - 深層 (L9-L17): 微量 LoRA → 輕微調整
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        shallow_adapter_positions: List[int] = [2, 4],
        mid_adapter_positions: List[int] = [6],
        mid_lora_layers: List[int] = [5, 6, 7, 8],
        deep_lora_layers: List[int] = [9, 10, 11, 12, 13, 14, 15, 16],
        shallow_reduction: int = 2,
        mid_reduction: int = 4,
        mid_lora_rank: int = 32,
        deep_lora_rank: int = 16,
        dropout: float = 0.1,
        init_scale: float = 0.01,
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

        # Student: Adapter + LoRA
        print("=" * 60)
        print("Loading Student with Adapter + LoRA...")
        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        # 凍結 Student 的所有原始權重
        for param in self.student.parameters():
            param.requires_grad = False

        # 替換 encoder
        original_encoder = self.student.feature_extractor.encodec.encoder
        self.student.feature_extractor.encodec.encoder = EncoderWithAdapterLoRA(
            original_encoder=original_encoder,
            shallow_adapter_positions=shallow_adapter_positions,
            mid_adapter_positions=mid_adapter_positions,
            mid_lora_layers=mid_lora_layers,
            deep_lora_layers=deep_lora_layers,
            shallow_reduction=shallow_reduction,
            mid_reduction=mid_reduction,
            mid_lora_rank=mid_lora_rank,
            mid_lora_alpha=mid_lora_rank * 2,
            deep_lora_rank=deep_lora_rank,
            deep_lora_alpha=deep_lora_rank * 2,
            dropout=dropout,
            init_scale=init_scale,
        )

        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # Codebook
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        # 統計參數
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.student.parameters())

        print("=" * 60)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
        print("=" * 60)

    def _freeze_quantizer(self, model, name: str):
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """獲取所有可訓練參數"""
        return [p for p in self.student.parameters() if p.requires_grad]

    def get_adapter_scales(self) -> Dict[str, float]:
        """獲取 Adapter scales"""
        return self.student.feature_extractor.encodec.encoder.get_adapter_scales()

    def _get_codebook(self) -> torch.Tensor:
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        return quantizer.vq.layers[0].codebook.detach().clone()

    def _get_teacher_codebook(self) -> torch.Tensor:
        return self.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def _get_student_codebook(self) -> torch.Tensor:
        return self.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()
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

        # Teacher forward
        self.teacher.eval()
        with torch.no_grad():
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward
        student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)

        self.student.feature_extractor.encodec.quantizer.eval()
        with torch.no_grad():
            student_vq = self.student.feature_extractor.encodec.quantizer(
                student_encoder_out, frame_rate=75, bandwidth=0.075
            )
            student_codes = student_vq.codes

        return {
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
            'adapter_scales': self.get_adapter_scales(),
        }

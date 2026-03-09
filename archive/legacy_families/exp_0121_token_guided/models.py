"""
exp_0121_token_guided: Model Definitions

基於 Exp K 架構，支援:
1. 全層 LoRA (所有 18 層) - 預設
2. Layer-Selective LoRA (只在指定層)
3. 中間層監督 (L3, L4, L6) - 沿用 Exp K
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenize-self-supervised')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer

# Apply LoRA patch for WavTokenizer compatibility
from done.exp.lora_encoder_denoising.wavtok_lora_patch import apply_lora_patch
apply_lora_patch()


# ============================================================
# 層級定義 (18 層 Conv1d)
# ============================================================

ALL_18_LAYERS = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",           # L0
    "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",   # L1
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",   # L2
    "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",  # L3
    "feature_extractor.encodec.encoder.model.3.conv.conv",           # L4 (downsample)
    "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",   # L5 ← 噪音敏感
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",   # L6 ← 噪音敏感
    "feature_extractor.encodec.encoder.model.4.shortcut.conv.conv",  # L7
    "feature_extractor.encodec.encoder.model.6.conv.conv",           # L8 (downsample) ← 噪音敏感
    "feature_extractor.encodec.encoder.model.7.block.1.conv.conv",   # L9
    "feature_extractor.encodec.encoder.model.7.block.3.conv.conv",   # L10
    "feature_extractor.encodec.encoder.model.7.shortcut.conv.conv",  # L11
    "feature_extractor.encodec.encoder.model.9.conv.conv",           # L12 (downsample)
    "feature_extractor.encodec.encoder.model.10.block.1.conv.conv",  # L13
    "feature_extractor.encodec.encoder.model.10.block.3.conv.conv",  # L14
    "feature_extractor.encodec.encoder.model.10.shortcut.conv.conv", # L15
    "feature_extractor.encodec.encoder.model.12.conv.conv",          # L16 (downsample)
    "feature_extractor.encodec.encoder.model.15.conv.conv",          # L17 (output)
]

# 預設的 Layer-Selective LoRA 配置
# 基於 exp_1231_feature 分析: model[4] 和 model[6] 噪音敏感度最高
NOISE_SENSITIVE_LAYERS = [
    "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",   # L5: model[4].block.1
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",   # L6: model[4].block.3
    "feature_extractor.encodec.encoder.model.6.conv.conv",           # L8: model[6] (Downsample2)
]


def get_target_layers(target_layer_patterns: List[str] = None) -> List[str]:
    """
    根據 pattern 匹配目標層

    Args:
        target_layer_patterns: 如 ["4.block.1", "4.block.3", "6.conv"]

    Returns:
        匹配的完整層名稱列表
    """
    if target_layer_patterns is None:
        return ALL_18_LAYERS

    matched_layers = []
    for layer_name in ALL_18_LAYERS:
        for pattern in target_layer_patterns:
            if pattern in layer_name:
                matched_layers.append(layer_name)
                break

    return matched_layers if matched_layers else ALL_18_LAYERS


class IntermediateExtractor(nn.Module):
    """
    提取 encoder 中間層輸出的包裝器
    沿用 Exp K 架構
    """

    def __init__(self, encoder: nn.Module, extract_indices: List[int]):
        """
        Args:
            encoder: 原始 encoder
            extract_indices: 要提取的層 index (model 的 index)
        """
        super().__init__()
        self.encoder = encoder
        self.extract_indices = sorted(extract_indices)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass，同時提取中間層輸出

        Args:
            x: (B, 1, T) 輸入音頻

        Returns:
            final_out: (B, C, T') 最終輸出
            intermediates: {index: (B, C, T'')} 中間層輸出
        """
        intermediates = {}

        for i, layer in enumerate(self.encoder.model):
            x = layer(x)

            if i in self.extract_indices:
                intermediates[i] = x

        return x, intermediates


class TeacherStudentTokenGuided(nn.Module):
    """
    Token-Guided LoRA 訓練模型

    基於 Exp K 架構:
    - 全層 LoRA 微調 (預設) 或 Layer-Selective LoRA
    - 支援中間層監督 (L3, L4, L6)
    - MSE + Triplet Loss (Exp K 驗證有效)
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        lora_dropout: float = 0.2,
        target_layer_patterns: List[str] = None,
        intermediate_indices: List[int] = None,
        device: str = "cuda",
    ):
        """
        Args:
            wavtok_config: WavTokenizer config path
            wavtok_ckpt: WavTokenizer checkpoint path
            lora_rank: LoRA rank (預設 256，沿用 Exp K)
            lora_alpha: LoRA alpha (預設 512，沿用 Exp K)
            lora_dropout: LoRA dropout
            target_layer_patterns: 要加 LoRA 的層 pattern，None = 全部 18 層
            intermediate_indices: 中間層監督位置，None = [3, 4, 6] (Exp K 預設)
            device: 計算裝置
        """
        super().__init__()
        self.device = device
        self.lora_rank = lora_rank

        # 預設中間層監督位置 (沿用 Exp K)
        if intermediate_indices is None:
            intermediate_indices = [3, 4, 6]  # model[3], model[4], model[6]
        self.intermediate_indices = intermediate_indices

        # 決定 LoRA 目標層
        target_modules = get_target_layers(target_layer_patterns)
        print(f"\n{'='*60}")
        print(f"Token-Guided LoRA Model (based on Exp K)")
        print(f"{'='*60}")
        print(f"  LoRA layers: {len(target_modules)} layers")
        if target_layer_patterns:
            print(f"  Patterns: {target_layer_patterns}")
        print(f"  Intermediate supervision: {intermediate_indices}")

        # Teacher: 完全凍結，加上中間層提取器
        print(f"\n{'='*60}")
        print("Loading Teacher (frozen) with intermediate extraction...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 包裝 teacher encoder 以提取中間層
        original_teacher_encoder = self.teacher.feature_extractor.encodec.encoder
        self.teacher_extractor = IntermediateExtractor(
            original_teacher_encoder,
            intermediate_indices
        )

        self.teacher = self.teacher.to(device)
        self._freeze_quantizer(self.teacher, "Teacher")

        # Student: LoRA 微調，加上中間層提取器
        print(f"\n{'='*60}")
        print(f"Loading Student with LoRA and intermediate extraction...")
        print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}, Dropout: {lora_dropout}")

        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()

        # 包裝 student encoder 以提取中間層
        original_student_encoder = self.student.feature_extractor.encodec.encoder
        self.student_extractor = IntermediateExtractor(
            original_student_encoder,
            intermediate_indices
        )

        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # 獲取 codebook 並保存初始狀態
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        print(f"\n{'='*60}")
        print(f"Codebook shape: {self.codebook.shape}")
        print(f"Intermediate supervision at layers: {intermediate_indices}")
        print(f"{'='*60}\n")

    def _freeze_quantizer(self, model, name: str):
        """凍結 quantizer，防止 codebook 漂移"""
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen")

    def _get_codebook(self) -> torch.Tensor:
        """獲取 codebook (4096, 512)"""
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        codebook = quantizer.vq.layers[0].codebook.detach().clone()
        return codebook

    def _get_teacher_codebook(self) -> torch.Tensor:
        return self.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def _get_student_codebook(self) -> torch.Tensor:
        return self.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def train(self, mode: bool = True):
        """確保 teacher 和 quantizer 永遠是 eval"""
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
                raise RuntimeError(f"Teacher codebook drift: {teacher_drift:.8f}")
            if student_drift > 1e-7:
                raise RuntimeError(f"Student codebook drift: {student_drift:.8f}")

        return result

    def forward(self, noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> dict:
        """
        Forward pass with intermediate extraction

        Args:
            noisy_audio: (B, T) 或 (B, 1, T) 帶噪音音頻
            clean_audio: (B, T) 或 (B, 1, T) 乾淨音頻

        Returns:
            dict with encoder outputs, codes, and intermediate features
        """
        # 確保維度正確 (B, 1, T)
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward with intermediate extraction
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            teacher_encoder_out, teacher_intermediates = self.teacher_extractor(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward with intermediate extraction
        student_encoder_out, student_intermediates = self.student_extractor(noisy_audio)

        self.student.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            quantizer = self.student.feature_extractor.encodec.quantizer
            student_vq = quantizer(student_encoder_out, frame_rate=75, bandwidth=0.075)
            student_codes = student_vq.codes

        return {
            'student_encoder_out': student_encoder_out,    # (B, 512, T')
            'teacher_encoder_out': teacher_encoder_out,    # (B, 512, T')
            'student_codes': student_codes,                # (B, 1, T')
            'teacher_codes': teacher_codes,                # (B, 1, T')
            'codebook': self.codebook,
            # 中間層輸出 (for intermediate supervision)
            'student_intermediates': student_intermediates,
            'teacher_intermediates': teacher_intermediates,
        }

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to audio

        Args:
            features: (B, 512, T') encoder output

        Returns:
            audio: (B, T) reconstructed audio
        """
        # WavTokenizer.decode() 需要 bandwidth_id 作為 kwarg (因為 backbone 使用 adanorm)
        bandwidth_id = torch.tensor([0], device=features.device)
        audio = self.teacher.decode(features, bandwidth_id=bandwidth_id)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return audio

"""
exp_0112_intermediate: Exp K - 中間層監督訓練

核心設計:
- 在 encoder 中間層 (L4, L8) 加入額外的監督信號
- 讓噪音敏感層直接獲得「應該輸出什麼」的指導
- 不需要等梯度從最終輸出一路傳回

架構:
┌─────────────────────────────────────────────────────────────┐
│  Teacher (Clean)           Student (Noisy)                  │
│                                                             │
│  Clean Audio               Noisy Audio                      │
│      ↓                         ↓                            │
│  [L0-L4] ──── Loss₁ ──── [L0-L4 + LoRA]                    │
│      ↓       (MSE)             ↓                            │
│  [L5-L8] ──── Loss₂ ──── [L5-L8 + LoRA]                    │
│      ↓       (MSE)             ↓                            │
│  [L9-L17] ─── Loss₃ ─── [L9-L17 + LoRA]                    │
│      ↓      (Final)            ↓                            │
│  Teacher Out              Student Out                       │
│                                                             │
│  Total Loss = λ₁×Loss₁ + λ₂×Loss₂ + Loss₃                  │
└─────────────────────────────────────────────────────────────┘

為什麼這樣設計:
1. 直接監督: 中層不用等梯度傳回，直接知道目標
2. 針對性: L4/L8 是噪音破壞最嚴重的區域
3. 多尺度學習: 每個階段都有明確的學習目標
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

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from exp_1201.wavtok_lora_patch import apply_lora_patch

apply_lora_patch()


class CodebookDriftError(Exception):
    """Codebook 漂移錯誤"""
    pass


# ============================================================
# 層級定義 (和 exp_0112 一致)
# ============================================================

ALL_18_LAYERS = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",           # L0
    "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",   # L1
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",   # L2
    "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",  # L3
    "feature_extractor.encodec.encoder.model.3.conv.conv",           # L4 (downsample) ← 監督點1
    "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",   # L5
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",   # L6
    "feature_extractor.encodec.encoder.model.4.shortcut.conv.conv",  # L7
    "feature_extractor.encodec.encoder.model.6.conv.conv",           # L8 (downsample) ← 監督點2
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

# 中間層監督位置 (model 的 index)
# L4 是 model[3], L8 是 model[6]
INTERMEDIATE_SUPERVISION_POINTS = {
    'L4': 3,   # 淺層輸出 (第一個 downsample)
    'L8': 6,   # 中層輸出 (第二個 downsample)
}


class IntermediateExtractor(nn.Module):
    """
    提取 encoder 中間層輸出的包裝器
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


class TeacherStudentIntermediate(nn.Module):
    """
    Exp K: 中間層監督訓練模型

    核心設計:
    - 全層 LoRA 微調 (保持梯度流通)
    - 在 L4, L8 加入中間層監督
    - 多尺度 Loss: 中間層 MSE + 最終輸出 Loss
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        lora_dropout: float = 0.2,
        intermediate_indices: List[int] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.lora_rank = lora_rank

        # 預設監督位置
        if intermediate_indices is None:
            intermediate_indices = [3, 6]  # L4, L8
        self.intermediate_indices = intermediate_indices

        # Teacher: 完全凍結，加上中間層提取器
        print("=" * 60)
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
        print(f"  Intermediate extraction at: {intermediate_indices}")

        # Student: LoRA 微調，加上中間層提取器
        print("=" * 60)
        print("Loading Student with LoRA and intermediate extraction...")
        print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}")

        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=ALL_18_LAYERS,
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()

        # 包裝 student encoder 以提取中間層
        # 注意: LoRA 已經應用，所以這裡的 encoder 已經帶有 LoRA
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

        print("=" * 60)
        print(f"Codebook shape: {self.codebook.shape}")
        print(f"Intermediate supervision at layers: {intermediate_indices}")
        print("=" * 60)

    def _freeze_quantizer(self, model, name: str):
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen")

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
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
            # 中間層輸出
            'student_intermediates': student_intermediates,
            'teacher_intermediates': teacher_intermediates,
        }

    def compute_ce_logits(self, encoder_out):
        B, C, T = encoder_out.shape
        z = encoder_out.permute(0, 2, 1)
        logits = 2 * torch.matmul(z, self.codebook.t())
        c_sq = (self.codebook ** 2).sum(dim=1)
        logits = logits - c_sq.unsqueeze(0).unsqueeze(0)
        logits = logits.permute(0, 2, 1)
        return logits


class IntermediateSupervisionLoss(nn.Module):
    """
    中間層監督 Loss

    計算中間層的 MSE Loss，用於直接監督噪音敏感層
    """

    def __init__(
        self,
        intermediate_weights: Dict[int, float] = None,
        reduction: str = 'mean',
    ):
        """
        Args:
            intermediate_weights: {layer_index: weight} 各層的權重
            reduction: 'mean' or 'sum'
        """
        super().__init__()

        if intermediate_weights is None:
            # 預設權重: L4 和 L8 各 0.5
            intermediate_weights = {3: 0.5, 6: 0.5}

        self.intermediate_weights = intermediate_weights
        self.reduction = reduction

    def forward(
        self,
        student_intermediates: Dict[int, torch.Tensor],
        teacher_intermediates: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算中間層監督 Loss

        Args:
            student_intermediates: {index: tensor} Student 中間層輸出
            teacher_intermediates: {index: tensor} Teacher 中間層輸出

        Returns:
            loss: 總 loss
            loss_info: 各層 loss 資訊
        """
        total_loss = 0.0
        loss_info = {}

        for idx, weight in self.intermediate_weights.items():
            if idx in student_intermediates and idx in teacher_intermediates:
                student_feat = student_intermediates[idx]
                teacher_feat = teacher_intermediates[idx]

                # 確保維度匹配 (可能因為 downsample 導致時間維度略有不同)
                min_t = min(student_feat.shape[-1], teacher_feat.shape[-1])
                student_feat = student_feat[..., :min_t]
                teacher_feat = teacher_feat[..., :min_t]

                layer_loss = F.mse_loss(student_feat, teacher_feat, reduction=self.reduction)
                total_loss = total_loss + weight * layer_loss

                loss_info[f'intermediate_L{idx}_loss'] = layer_loss.item()

        loss_info['intermediate_total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, loss_info

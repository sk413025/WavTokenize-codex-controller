"""
exp_1231_1: 凍結深層實驗

基於 exp_1231_feature 分析結論:
- 噪音主要影響 mid-level (L5-L6)
- 深層 (L13-L15) 對噪音魯棒
- 但 LoRA 訓練後深層變化最大，淺層變化最小
- 這與降噪任務需求相反

實驗策略:
- 只對淺/中層 (L0-L8) 加 LoRA
- 完全凍結深層 (L9-L17)
- 驗證假設: 深層過度變化是 Train/Val gap 的原因
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List, Optional

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
# Layer Groups 定義 (按照 ANALYSIS.md 分組)
# ============================================================

# L0: input
INPUT_LAYER = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",
]

# L1-L4: low_level
LOW_LEVEL_LAYERS = [
    "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",   # L1
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",   # L2
    "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",  # L3
    "feature_extractor.encodec.encoder.model.3.conv.conv",           # L4 (downsample)
]

# L5-L8: mid_level (噪音最敏感!)
MID_LEVEL_LAYERS = [
    "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",   # L5
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",   # L6
    "feature_extractor.encodec.encoder.model.4.shortcut.conv.conv",  # L7
    "feature_extractor.encodec.encoder.model.6.conv.conv",           # L8 (downsample)
]

# L9-L12: semantic
SEMANTIC_LAYERS = [
    "feature_extractor.encodec.encoder.model.7.block.1.conv.conv",   # L9
    "feature_extractor.encodec.encoder.model.7.block.3.conv.conv",   # L10
    "feature_extractor.encodec.encoder.model.7.shortcut.conv.conv",  # L11
    "feature_extractor.encodec.encoder.model.9.conv.conv",           # L12 (downsample)
]

# L13-L16: abstract (最魯棒!)
ABSTRACT_LAYERS = [
    "feature_extractor.encodec.encoder.model.10.block.1.conv.conv",  # L13
    "feature_extractor.encodec.encoder.model.10.block.3.conv.conv",  # L14
    "feature_extractor.encodec.encoder.model.10.shortcut.conv.conv", # L15
    "feature_extractor.encodec.encoder.model.12.conv.conv",          # L16 (downsample)
]

# L17: output
OUTPUT_LAYER = [
    "feature_extractor.encodec.encoder.model.15.conv.conv",
]

# ============================================================
# 實驗用 Layer Presets
# ============================================================

# 全部 18 層 (baseline)
ALL_18_LAYERS = (
    INPUT_LAYER + LOW_LEVEL_LAYERS + MID_LEVEL_LAYERS +
    SEMANTIC_LAYERS + ABSTRACT_LAYERS + OUTPUT_LAYER
)

# ★★★ Exp B: 淺/中層 only (L0-L8) - 凍結深層 ★★★
SHALLOW_MID_LAYERS = INPUT_LAYER + LOW_LEVEL_LAYERS + MID_LEVEL_LAYERS

# 噪音敏感層 (L5-L6 為核心)
NOISE_SENSITIVE_LAYERS = MID_LEVEL_LAYERS

# 深層 only (L9-L17) - 用於對照
DEEP_LAYERS = SEMANTIC_LAYERS + ABSTRACT_LAYERS + OUTPUT_LAYER

# Layer presets
LORA_LAYER_PRESETS = {
    'all_18': ALL_18_LAYERS,
    'shallow_mid': SHALLOW_MID_LAYERS,     # ★ Exp B: L0-L8
    'noise_sensitive': NOISE_SENSITIVE_LAYERS,  # L5-L8 only
    'deep_only': DEEP_LAYERS,              # L9-L17 only (對照)
}


class TeacherStudentShallowLoRA(nn.Module):
    """
    淺層 LoRA 的 Teacher-Student 模型

    核心改動:
    - 只對淺/中層 (L0-L8) 加 LoRA
    - 深層 (L9-L17) 完全凍結
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        lora_dropout: float = 0.2,
        lora_layers: str = 'shallow_mid',  # 預設使用淺/中層
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # 解析 LoRA 層配置
        if isinstance(lora_layers, str):
            if lora_layers in LORA_LAYER_PRESETS:
                target_modules = LORA_LAYER_PRESETS[lora_layers]
            else:
                raise ValueError(f"Unknown lora_layers preset: {lora_layers}. "
                               f"Available: {list(LORA_LAYER_PRESETS.keys())}")
        else:
            target_modules = lora_layers

        self.target_modules = target_modules
        self.lora_layers_name = lora_layers if isinstance(lora_layers, str) else 'custom'

        # Teacher: 完全凍結
        print("=" * 60)
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)
        self._freeze_quantizer(self.teacher, "Teacher")

        # Student: 只對指定層加 LoRA
        print("=" * 60)
        print(f"Loading Student with LoRA...")
        print(f"  Preset: {self.lora_layers_name}")
        print(f"  Target layers: {len(target_modules)} / 18")
        print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}")

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
        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # 獲取 codebook 並保存初始狀態
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        print("=" * 60)
        print(f"Codebook shape: {self.codebook.shape}")
        self._print_layer_coverage()
        print("=" * 60)

    def _print_layer_coverage(self):
        """打印層覆蓋情況"""
        all_layers = ALL_18_LAYERS
        covered = set(self.target_modules)

        print("\nLayer coverage:")
        layer_groups = [
            ("Input (L0)", INPUT_LAYER),
            ("Low-level (L1-L4)", LOW_LEVEL_LAYERS),
            ("Mid-level (L5-L8)", MID_LEVEL_LAYERS),
            ("Semantic (L9-L12)", SEMANTIC_LAYERS),
            ("Abstract (L13-L16)", ABSTRACT_LAYERS),
            ("Output (L17)", OUTPUT_LAYER),
        ]

        for group_name, layers in layer_groups:
            n_covered = sum(1 for l in layers if l in covered)
            status = "✓ LoRA" if n_covered == len(layers) else (
                "◐ Partial" if n_covered > 0 else "✗ Frozen"
            )
            print(f"  {group_name}: {status} ({n_covered}/{len(layers)})")

    def _freeze_quantizer(self, model, name: str):
        """凍結 quantizer"""
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen")

    def _get_codebook(self) -> torch.Tensor:
        """從 Teacher 獲取 codebook"""
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        codebook = quantizer.vq.layers[0].codebook.detach().clone()
        return codebook

    def _get_teacher_codebook(self) -> torch.Tensor:
        return self.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def _get_student_codebook(self) -> torch.Tensor:
        return self.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def train(self, mode: bool = True):
        """覆寫 train 方法，確保 quantizer 始終 eval"""
        super().train(mode)
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()
        self.student.feature_extractor.encodec.quantizer.eval()
        return self

    def check_codebook_integrity(self, raise_error: bool = True) -> dict:
        """檢查 codebook 是否被意外修改"""
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
        """前向傳播"""
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

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
            quantizer = self.student.feature_extractor.encodec.quantizer
            student_vq = quantizer(student_encoder_out, frame_rate=75, bandwidth=0.075)
            student_codes = student_vq.codes

        return {
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
        }

    def compute_ce_logits(self, encoder_out):
        """計算 Cross-Entropy logits"""
        B, C, T = encoder_out.shape
        z = encoder_out.permute(0, 2, 1)
        logits = 2 * torch.matmul(z, self.codebook.t())
        c_sq = (self.codebook ** 2).sum(dim=1)
        logits = logits - c_sq.unsqueeze(0).unsqueeze(0)
        logits = logits.permute(0, 2, 1)
        return logits

"""
exp_1217: 可配置 LoRA 層的模型

支持:
1. 選擇性 LoRA 層 (18層全部 vs 關鍵8層)
2. 可配置 LoRA rank
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from exp_1201.wavtok_lora_patch import apply_lora_patch

apply_lora_patch()


class CodebookDriftError(Exception):
    """Codebook 漂移錯誤"""
    pass


# 全部 18 個 encoder conv 層
ALL_ENCODER_CONV_MODULES = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",
    "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",
    "feature_extractor.encodec.encoder.model.3.conv.conv",
    "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.4.shortcut.conv.conv",
    "feature_extractor.encodec.encoder.model.6.conv.conv",
    "feature_extractor.encodec.encoder.model.7.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.7.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.7.shortcut.conv.conv",
    "feature_extractor.encodec.encoder.model.9.conv.conv",
    "feature_extractor.encodec.encoder.model.10.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.10.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.10.shortcut.conv.conv",
    "feature_extractor.encodec.encoder.model.12.conv.conv",
    "feature_extractor.encodec.encoder.model.15.conv.conv",
]

# 關鍵 8 層 (輸入/輸出投影 + downsampling + 關鍵 residual)
# 注意: 這個配置遺漏了 model.7 和 model.10 的語義層！
CRITICAL_ENCODER_CONV_MODULES = [
    # 輸入投影
    "feature_extractor.encodec.encoder.model.0.conv.conv",
    # 各 downsampling block 的主卷積 (stride > 1)
    "feature_extractor.encodec.encoder.model.3.conv.conv",
    "feature_extractor.encodec.encoder.model.6.conv.conv",
    "feature_extractor.encodec.encoder.model.9.conv.conv",
    "feature_extractor.encodec.encoder.model.12.conv.conv",
    # Residual block 的輸出卷積 (信息瓶頸)
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",
    # 輸出投影
    "feature_extractor.encodec.encoder.model.15.conv.conv",
]

# 關鍵 10 層 (修正版: 包含 model.7 和 model.10 語義層)
# 設計原則:
# - model.0: 輸入投影 (首層處理噪聲)
# - model.7.*: 語義提取 ResBlock ★★★ (關鍵!)
# - model.10.*: 高階抽象 ResBlock ★★★ (關鍵!)
# - model.15: 輸出投影到 VQ 空間
# - Downsample 層保留以維持空間一致性
CRITICAL_10_ENCODER_CONV_MODULES = [
    # 輸入投影
    "feature_extractor.encodec.encoder.model.0.conv.conv",
    # Downsample 層 (維持空間結構)
    "feature_extractor.encodec.encoder.model.3.conv.conv",
    "feature_extractor.encodec.encoder.model.6.conv.conv",
    # model.7 語義提取 ResBlock (完整覆蓋) ★★★
    "feature_extractor.encodec.encoder.model.7.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.7.block.3.conv.conv",
    # Downsample 3
    "feature_extractor.encodec.encoder.model.9.conv.conv",
    # model.10 高階抽象 ResBlock (完整覆蓋) ★★★
    "feature_extractor.encodec.encoder.model.10.block.1.conv.conv",
    "feature_extractor.encodec.encoder.model.10.block.3.conv.conv",
    # Downsample 4
    "feature_extractor.encodec.encoder.model.12.conv.conv",
    # 輸出投影
    "feature_extractor.encodec.encoder.model.15.conv.conv",
]

# 層配置預設
LORA_LAYER_PRESETS = {
    'all_18': ALL_ENCODER_CONV_MODULES,
    'critical_8': CRITICAL_ENCODER_CONV_MODULES,
    'critical_10': CRITICAL_10_ENCODER_CONV_MODULES,
}


class TeacherStudentConfigurableLoRA(nn.Module):
    """
    可配置 LoRA 層的 Teacher-Student 模型

    支持:
    1. 選擇性 LoRA 層 (all_18 vs critical_8)
    2. 可配置 LoRA rank
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.1,
        lora_layers: str = 'all_18',  # 'all_18' or 'critical_8' or custom list
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # 解析 LoRA 層配置
        if isinstance(lora_layers, str):
            if lora_layers in LORA_LAYER_PRESETS:
                target_modules = LORA_LAYER_PRESETS[lora_layers]
            else:
                raise ValueError(f"Unknown lora_layers preset: {lora_layers}")
        else:
            target_modules = lora_layers

        self.target_modules = target_modules

        # Teacher: 完全凍結
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)

        # 凍結 Teacher quantizer
        self._freeze_quantizer(self.teacher, "Teacher")

        # Student: LoRA
        print(f"Loading Student with LoRA (rank={lora_rank}, {len(target_modules)} layers)...")
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

        # 凍結 Student quantizer
        self._freeze_quantizer(self.student, "Student")

        # 獲取 codebook 並保存初始狀態
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()
        print(f"Codebook shape: {self.codebook.shape}")

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

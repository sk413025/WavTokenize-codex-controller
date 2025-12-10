"""
exp_1210: 修復版模型定義

修復問題:
1. Teacher 意外進入 train 模式 - 覆寫 train() 方法
2. Codebook EMA 漂移 - 凍結所有 quantizer
3. 添加 codebook 安全檢查

架構:
    Noisy Audio → Encoder(LoRA 18層) → VQ(凍結) → tokens
    Clean Audio → Encoder(凍結) → VQ(凍結) → tokens (Teacher)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from exp_1201.wavtok_lora_patch import apply_lora_patch

apply_lora_patch()


class CodebookDriftError(Exception):
    """Codebook 漂移錯誤"""
    pass


class TeacherStudentExpandedLoRA(nn.Module):
    """
    修復版 Teacher-Student 模型

    修復:
    1. 覆寫 train() 方法，確保 Teacher 始終 eval
    2. 凍結 Teacher 和 Student 的 quantizer，防止 EMA 更新
    3. 添加 codebook 安全檢查

    架構:
        Noisy Audio → Encoder(LoRA 18層) → VQ(凍結) → tokens
        Clean Audio → Encoder(凍結) → VQ(凍結) → tokens (Teacher)
    """

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

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

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
        print(f"Loading Student with Expanded LoRA (rank={lora_rank}, {len(self.ALL_ENCODER_CONV_MODULES)} layers)...")
        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self.ALL_ENCODER_CONV_MODULES,
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()
        self.student = self.student.to(device)

        # 凍結 Student quantizer
        self._freeze_quantizer(self.student, "Student")

        # 獲取 codebook 並保存初始狀態（用於安全檢查）
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()
        print(f"Codebook shape: {self.codebook.shape}")

    def _freeze_quantizer(self, model, name: str):
        """凍結 quantizer，防止 EMA 更新"""
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen (eval mode, requires_grad=False)")

    def _get_codebook(self) -> torch.Tensor:
        """從 Teacher 獲取 codebook"""
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        codebook = quantizer.vq.layers[0].codebook.detach().clone()
        return codebook

    def _get_teacher_codebook(self) -> torch.Tensor:
        """獲取 Teacher 當前 codebook"""
        return self.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def _get_student_codebook(self) -> torch.Tensor:
        """獲取 Student 當前 codebook"""
        return self.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def train(self, mode: bool = True):
        """
        覆寫 train 方法

        確保:
        1. Teacher 始終保持 eval 模式
        2. 所有 quantizer 始終保持 eval 模式
        """
        super().train(mode)

        # Teacher 始終 eval
        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        # Student quantizer 也要 eval（防止 EMA 更新）
        self.student.feature_extractor.encodec.quantizer.eval()

        return self

    def check_codebook_integrity(self, raise_error: bool = True) -> dict:
        """
        檢查 codebook 是否被意外修改

        Args:
            raise_error: 如果檢測到漂移，是否拋出錯誤

        Returns:
            dict 包含檢查結果
        """
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
                raise CodebookDriftError(
                    f"Teacher codebook 漂移! drift={teacher_drift:.8f}\n"
                    f"這表示 Teacher quantizer 意外進入了 train 模式"
                )
            if student_drift > 1e-7:
                raise CodebookDriftError(
                    f"Student codebook 漂移! drift={student_drift:.8f}\n"
                    f"這表示 Student quantizer 意外進入了 train 模式"
                )

        return result

    def forward(self, noisy_audio: torch.Tensor, clean_audio: torch.Tensor) -> dict:
        """
        前向傳播

        Args:
            noisy_audio: 帶噪音的音頻
            clean_audio: 乾淨的音頻

        Returns:
            dict 包含各種輸出和 codebook
        """
        # 確保格式
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward (確保 eval 模式)
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

        # VQ (確保 eval 模式)
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

"""
exp_1222: 支援 Audio Domain Loss 的模型

核心變化:
1. forward() 支援直接 decode encoder features (bypass VQ)
2. 訓練時使用連續 features，推論時可選擇 VQ 離散化
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')

from peft import LoraConfig, get_peft_model
from decoder.pretrained import WavTokenizer
from exp_1201.wavtok_lora_patch import apply_lora_patch
from exp_1217.models import (
    ALL_ENCODER_CONV_MODULES,
    CRITICAL_ENCODER_CONV_MODULES,
    CRITICAL_10_ENCODER_CONV_MODULES,
    LORA_LAYER_PRESETS,
    CodebookDriftError,
)

apply_lora_patch()


class TeacherStudentAudioLoss(nn.Module):
    """
    支援 Audio Domain Loss 的 Teacher-Student 模型

    訓練流程:
    1. Student Encoder(noisy) → features
    2. features → Decoder → denoised audio (bypass VQ)
    3. Audio Loss(denoised, clean)

    推論流程 (可選):
    1. Student Encoder(noisy) → features
    2. features → VQ → tokens
    3. tokens → Decoder → denoised audio
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.1,
        lora_layers: str = 'all_18',
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

        # Teacher: 完全凍結 (用於生成 ground truth)
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)
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
        self._freeze_quantizer(self.student, "Student")

        # Codebook
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

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        直接將 encoder features decode 為音頻 (bypass VQ)

        Args:
            features: (B, C, L) encoder output

        Returns:
            audio: (B, T) reconstructed audio
        """
        bandwidth_id = torch.tensor([0], device=features.device)

        # 使用 teacher 的 decoder (與 student 共享權重)
        audio = self.teacher.decode(features, bandwidth_id=bandwidth_id)

        if audio.dim() == 3:
            audio = audio.squeeze(1)

        return audio

    def forward(
        self,
        noisy_audio: torch.Tensor,
        clean_audio: torch.Tensor,
        return_audio: bool = True,
    ) -> dict:
        """
        前向傳播

        Args:
            noisy_audio: (B, T) or (B, 1, T) noisy input
            clean_audio: (B, T) or (B, 1, T) clean target
            return_audio: 是否返回 decoded audio (用於 audio loss)

        Returns:
            dict with:
                - student_encoder_out: (B, C, L) student encoder features
                - teacher_encoder_out: (B, C, L) teacher encoder features
                - student_codes: (1, B, L) student VQ codes
                - teacher_codes: (1, B, L) teacher VQ codes
                - codebook: (V, C)
                - denoised_audio: (B, T) if return_audio=True
                - teacher_recon_audio: (B, T) if return_audio=True
        """
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward (frozen)
        self.teacher.eval()
        with torch.no_grad():
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

        # Student forward
        student_encoder_out = self.student.feature_extractor.encodec.encoder(noisy_audio)

        # VQ for codes (with no_grad, just for monitoring)
        with torch.no_grad():
            student_vq = self.student.feature_extractor.encodec.quantizer(
                student_encoder_out.detach(), frame_rate=75, bandwidth=0.075
            )
            student_codes = student_vq.codes

        result = {
            'student_encoder_out': student_encoder_out,
            'teacher_encoder_out': teacher_encoder_out,
            'student_codes': student_codes,
            'teacher_codes': teacher_codes,
            'codebook': self.codebook,
        }

        # Decode features to audio (bypass VQ) - 這是關鍵！
        if return_audio:
            # Student features → audio (for audio loss)
            denoised_audio = self.decode_features(student_encoder_out)
            result['denoised_audio'] = denoised_audio

            # Teacher features → audio (as reference)
            with torch.no_grad():
                teacher_recon_audio = self.decode_features(teacher_encoder_out)
            result['teacher_recon_audio'] = teacher_recon_audio

        return result

    def inference(
        self,
        noisy_audio: torch.Tensor,
        use_vq: bool = False,
    ) -> torch.Tensor:
        """
        推論模式

        Args:
            noisy_audio: (B, T) input
            use_vq: 是否使用 VQ 離散化

        Returns:
            denoised_audio: (B, T)
        """
        self.eval()

        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)

        with torch.no_grad():
            # Encode
            features = self.student.feature_extractor.encodec.encoder(noisy_audio)

            if use_vq:
                # 經過 VQ
                vq_result = self.student.feature_extractor.encodec.quantizer(
                    features, frame_rate=75, bandwidth=0.075
                )
                # codes → features
                codes = vq_result.codes  # (1, B, L)
                features = self.teacher.codes_to_features(codes)

            # Decode
            audio = self.decode_features(features)

        return audio

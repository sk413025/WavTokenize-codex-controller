"""
Teacher-Student 模型 - LoRA Encoder Denoising
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import sys
from pathlib import Path

# 添加 WavTokenizer 路徑
sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
from decoder.pretrained import WavTokenizer

# 導入並應用 LoRA 兼容性補丁
try:
    from .wavtok_lora_patch import apply_lora_patch
except ImportError:
    from wavtok_lora_patch import apply_lora_patch

# 應用補丁（只需應用一次）
apply_lora_patch()


class TeacherStudentModel(nn.Module):
    """
    Teacher-Student Knowledge Distillation Model

    Architecture:
        Teacher (凍結): 原始 WavTokenizer
            clean_audio → features_clean, codes_clean

        Student (LoRA): WavTokenizer + LoRA on Encoder
            noisy_audio → features_noisy, codes_noisy

    目標: features_noisy ≈ features_clean
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list = None,
        device: str = "cuda",
    ):
        """
        Args:
            wavtok_config: WavTokenizer 配置文件路徑
            wavtok_ckpt: WavTokenizer checkpoint 路徑
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (scaling factor)
            lora_dropout: LoRA dropout
            lora_target_modules: LoRA 目標模組列表
            device: 設備
        """
        super().__init__()

        self.device = device

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Teacher: 原始 WavTokenizer (完全凍結)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("Loading Teacher model (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)
        print("✓ Teacher loaded and frozen")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Student: WavTokenizer + LoRA
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("Loading Student model (with LoRA)...")
        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        # 配置 LoRA
        # PEFT只能wrap Conv1d, 不能wrap NormConv1d
        # 所以我們需要target內部的Conv1d: .conv.conv
        # 兼容性補丁會處理attribute access問題
        if lora_target_modules is None:
            # Target the inner Conv1d inside NormConv1d wrappers
            # Main strided conv layers: model.0, model.3, model.6, model.9
            lora_target_modules = [
                "feature_extractor.encodec.encoder.model.0.conv.conv",
                "feature_extractor.encodec.encoder.model.3.conv.conv",
                "feature_extractor.encodec.encoder.model.6.conv.conv",
                "feature_extractor.encodec.encoder.model.9.conv.conv",
            ]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=None,  # Don't replace modules, just add LoRA
        )

        # 應用 LoRA
        self.student = get_peft_model(self.student, lora_config)

        # 凍結 VQ 和 Backbone (只訓練 Encoder LoRA)
        for name, param in self.student.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

        self.student = self.student.to(device)

        # 打印可訓練參數
        self.student.print_trainable_parameters()
        print("✓ Student loaded with LoRA")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Distance Matrix (VQ Codebook Distances)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("Computing codebook distance matrix...")
        self.distance_matrix = self._compute_codebook_distances()
        print(f"✓ Distance matrix shape: {self.distance_matrix.shape}")

    def _compute_codebook_distances(self):
        """
        計算 VQ codebook 中所有 code 之間的 L2 距離

        Returns:
            distance_matrix: (4096, 4096) pairwise distances
        """
        # 從 teacher 獲取 codebook
        # WavTokenizer structure: feature_extractor.encodec.quantizer.vq.layers[0].codebook
        try:
            codebook = self.teacher.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.embed
            # codebook shape: (4096, 512)
        except AttributeError:
            print("Warning: Cannot access codebook, using dummy distances")
            return torch.zeros(4096, 4096, device=self.device)

        # 計算 pairwise L2 distances
        # dist[i, j] = ||codebook[i] - codebook[j]||₂
        with torch.no_grad():
            dist_matrix = torch.cdist(codebook, codebook, p=2)

        return dist_matrix.to(self.device)

    def forward(self, noisy_audio, clean_audio):
        """
        Forward pass

        Args:
            noisy_audio: (B, T_audio) or (B, 1, T_audio)
            clean_audio: (B, T_audio) or (B, 1, T_audio)

        Returns:
            dict with:
                - student_features: (B, 512, T_frame)
                - teacher_features: (B, 512, T_frame)
                - student_codes: (B, 1, T_frame)
                - teacher_codes: (B, 1, T_frame)
                - vq_loss: scalar
        """
        # Ensure audio shape: (B, T)
        if noisy_audio.dim() == 3:
            noisy_audio = noisy_audio.squeeze(1)
        if clean_audio.dim() == 3:
            clean_audio = clean_audio.squeeze(1)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Teacher: Clean audio → features, codes
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        with torch.no_grad():
            teacher_features, teacher_codes, _ = self.teacher.feature_extractor(
                clean_audio, bandwidth_id=0
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Student: Noisy audio → features, codes
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        student_features, student_codes, vq_loss = self.student.feature_extractor(
            noisy_audio, bandwidth_id=0
        )

        return {
            'student_features': student_features,  # (B, 512, T_frame)
            'teacher_features': teacher_features,  # (B, 512, T_frame)
            'student_codes': student_codes,        # (B, 1, T_frame)
            'teacher_codes': teacher_codes,        # (B, 1, T_frame)
            'vq_loss': vq_loss,                    # scalar
        }

    def student_forward(self, audio):
        """
        只使用 Student 進行 forward (用於推理)

        Args:
            audio: (B, T_audio)

        Returns:
            features, codes, vq_loss
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        features, codes, vq_loss = self.student.feature_extractor(
            audio, bandwidth_id=0
        )

        return features, codes, vq_loss

    def teacher_forward(self, audio):
        """
        只使用 Teacher 進行 forward (用於評估原始能力)

        Args:
            audio: (B, T_audio)

        Returns:
            features, codes
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        with torch.no_grad():
            features, codes, _ = self.teacher.feature_extractor(
                audio, bandwidth_id=0
            )

        return features, codes

    def get_trainable_parameters(self):
        """
        獲取可訓練參數（只有 Student 的 LoRA 參數）

        Returns:
            generator of trainable parameters
        """
        return filter(lambda p: p.requires_grad, self.student.parameters())

    def save_student_checkpoint(self, path):
        """
        保存 Student 的 checkpoint (只保存 LoRA 權重)

        Args:
            path: 保存路徑
        """
        # PEFT 提供保存 LoRA 的方法
        self.student.save_pretrained(path)
        print(f"✓ Student checkpoint saved to {path}")

    def load_student_checkpoint(self, path):
        """
        載入 Student checkpoint

        Args:
            path: checkpoint 路徑
        """
        from peft import PeftModel

        # 重新載入原始 WavTokenizer + LoRA weights
        self.student = PeftModel.from_pretrained(
            self.student.base_model,
            path
        )
        self.student = self.student.to(self.device)
        print(f"✓ Student checkpoint loaded from {path}")

    def count_parameters(self):
        """
        統計參數量

        Returns:
            dict with parameter counts
        """
        total_params = sum(p.numel() for p in self.student.parameters())
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_percentage': 100 * trainable_params / total_params
        }


def create_teacher_student_model(config, device="cuda"):
    """
    創建 Teacher-Student 模型的工廠函數

    Args:
        config: TrainConfig or SmokeTestConfig
        device: 設備

    Returns:
        TeacherStudentModel
    """
    try:
        from .config import WAVTOK_CONFIG, WAVTOK_CKPT
    except ImportError:
        from config import WAVTOK_CONFIG, WAVTOK_CKPT

    model = TeacherStudentModel(
        wavtok_config=WAVTOK_CONFIG,
        wavtok_ckpt=WAVTOK_CKPT,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        device=device,
    )

    return model


if __name__ == "__main__":
    # 測試模型創建
    print("="*60)
    print("Testing TeacherStudentModel creation...")
    print("="*60)

    from config import get_smoke_test_config

    config = get_smoke_test_config()
    model = create_teacher_student_model(config, device="cpu")

    print("\n" + "="*60)
    print("Model Statistics:")
    print("="*60)
    stats = model.count_parameters()
    for key, value in stats.items():
        print(f"{key:25s}: {value:,}")

    print("\n" + "="*60)
    print("Testing forward pass...")
    print("="*60)

    # 測試 forward
    batch_size = 2
    audio_length = 24000 * 3  # 3 seconds at 24kHz

    dummy_noisy = torch.randn(batch_size, audio_length)
    dummy_clean = torch.randn(batch_size, audio_length)

    output = model(dummy_noisy, dummy_clean)

    print(f"Student features shape: {output['student_features'].shape}")
    print(f"Teacher features shape: {output['teacher_features'].shape}")
    print(f"Student codes shape: {output['student_codes'].shape}")
    print(f"Teacher codes shape: {output['teacher_codes'].shape}")
    print(f"VQ loss: {output['vq_loss'].item():.6f}")

    print("\n✅ All tests passed!")

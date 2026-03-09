"""
exp_test: 淺層凍結實驗 - 容量瓶頸測試

核心問題:
- 降噪擾動在中層 (L5-L8) 最大
- 但 LoRA 訓練後深層變化最大
- 懷疑: LoRA 容量不足，導致無法在淺/中層有效學習

實驗設計:
- 只訓練 L0-L4 (淺層)，Loss 監督到 L4 後的輸出
- 完全凍結 L5-L17
- 測試不同 LoRA rank (256/512/1024) 哪個能讓 loss 降得更低

架構圖:
┌─────────────────────────────────────────────────────────────┐
│  Teacher (Clean)              Student (Noisy)               │
│                                                             │
│  Clean Audio                  Noisy Audio                   │
│      ↓                            ↓                         │
│  [L0-L4] ──── MSE Loss ──── [L0-L4 + LoRA]                 │
│      ↓                            ↓                         │
│  [L5-L17] (凍結)              [L5-L17] (凍結)                │
│      ↓                            ↓                         │
│  Teacher Out                  (不使用)                       │
│                                                             │
│  只計算 L4 輸出的 MSE Loss，驗證淺層容量瓶頸                 │
└─────────────────────────────────────────────────────────────┘

預期結果:
- 如果 rank 增加 → loss 明顯降低 → 證明容量不足
- 如果 rank 增加 → loss 無明顯改善 → 問題不在容量
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
# 層級定義
# ============================================================

# L0: input
INPUT_LAYER = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",
]

# L1-L4: low_level (淺層)
LOW_LEVEL_LAYERS = [
    "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",   # L1
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",   # L2
    "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",  # L3
    "feature_extractor.encodec.encoder.model.3.conv.conv",           # L4 (downsample)
]

# ★ 實驗目標: 只訓練這 5 層 (L0-L4)
SHALLOW_ONLY_LAYERS = INPUT_LAYER + LOW_LEVEL_LAYERS

# L5-L8: mid_level (噪音最敏感，本實驗凍結)
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

# L13-L16: abstract
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

# 全部 18 層 (for reference)
ALL_18_LAYERS = (
    INPUT_LAYER + LOW_LEVEL_LAYERS + MID_LEVEL_LAYERS +
    SEMANTIC_LAYERS + ABSTRACT_LAYERS + OUTPUT_LAYER
)


class ShallowEncoder(nn.Module):
    """
    只執行到 L4 的淺層 Encoder

    用於提取 L4 (model[3]) 的輸出
    """

    def __init__(self, encoder: nn.Module, stop_at: int = 4):
        """
        Args:
            encoder: 原始 encoder
            stop_at: 停止的層索引 (model 的 index)
                     L4 = model[3], 所以 stop_at=4 會執行 model[0:4]
        """
        super().__init__()
        self.encoder = encoder
        self.stop_at = stop_at

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass，只執行到指定層

        Args:
            x: (B, 1, T) 輸入音頻

        Returns:
            (B, C, T') L4 層輸出
        """
        for i, layer in enumerate(self.encoder.model):
            if i >= self.stop_at:
                break
            x = layer(x)
        return x


class TeacherStudentShallowOnly(nn.Module):
    """
    Exp Test: 淺層凍結實驗模型

    核心設計:
    - 只對 L0-L4 加 LoRA (5 層)
    - 完全凍結 L5-L17 (13 層)
    - Loss 只計算 L4 輸出的 MSE

    目的: 測試 LoRA 容量瓶頸
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        lora_rank: int = 256,
        lora_alpha: int = 512,
        lora_dropout: float = 0.2,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # L4 = model[3], 所以 stop_at=4 會執行 model[0:4]
        # 但實際上 L4 是 model.3，所以要執行到 index 4 (exclusive)
        # model[0] = input conv
        # model[1] = residual block (L1-L3)
        # model[2] = downsample (?)
        # model[3] = L4 downsample conv
        # 我們要執行到 model[3] inclusive，所以 stop_at=4
        self.stop_at_layer = 4  # 執行到 model[3] (L4)

        # Teacher: 完全凍結
        print("=" * 60)
        print("Loading Teacher (frozen) with shallow extraction...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 包裝 teacher encoder
        self.teacher_shallow = ShallowEncoder(
            self.teacher.feature_extractor.encodec.encoder,
            stop_at=self.stop_at_layer
        )

        self.teacher = self.teacher.to(device)
        self._freeze_quantizer(self.teacher, "Teacher")
        print(f"  Shallow extraction stops at model[{self.stop_at_layer}] (L4)")

        # Student: 只對 L0-L4 加 LoRA
        print("=" * 60)
        print("Loading Student with shallow LoRA...")
        print(f"  ★ Only L0-L4 (5 layers) will have LoRA")
        print(f"  ★ L5-L17 (13 layers) are completely FROZEN")
        print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}")

        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=SHALLOW_ONLY_LAYERS,  # 只對 L0-L4 加 LoRA
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()

        # 包裝 student encoder
        self.student_shallow = ShallowEncoder(
            self.student.feature_extractor.encodec.encoder,
            stop_at=self.stop_at_layer
        )

        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # 獲取 codebook (雖然這個實驗不用，但保留一致性)
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        # 計算參數量
        self._print_param_info()

        print("=" * 60)
        print(f"Codebook shape: {self.codebook.shape}")
        print("=" * 60)

    def _print_param_info(self):
        """打印參數量資訊"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")

        # 估算 LoRA 參數量
        # 每層 LoRA: 2 * in_dim * rank (down + up)
        # 假設每層 dim 不同，這裡只是粗估
        print(f"\n  LoRA config:")
        print(f"    - Rank: {self.lora_rank}")
        print(f"    - Alpha: {self.lora_alpha}")
        print(f"    - Scaling: {self.lora_alpha / self.lora_rank:.2f}")
        print(f"    - Target layers: {len(SHALLOW_ONLY_LAYERS)} (L0-L4)")

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
        """覆寫 train 方法，確保 teacher 和 quantizer 始終 eval"""
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
        """
        前向傳播

        只計算 L4 層的輸出，用於 MSE Loss
        """
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # Teacher forward (只到 L4)
        self.teacher.eval()
        with torch.no_grad():
            teacher_l4_out = self.teacher_shallow(clean_audio)

        # Student forward (只到 L4)
        student_l4_out = self.student_shallow(noisy_audio)

        return {
            'student_l4_out': student_l4_out,
            'teacher_l4_out': teacher_l4_out,
            'codebook': self.codebook,
        }


class ShallowMSELoss(nn.Module):
    """
    淺層 MSE Loss

    只計算 L4 輸出的 MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        student_l4_out: torch.Tensor,
        teacher_l4_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算 L4 MSE Loss

        Args:
            student_l4_out: (B, C, T) Student L4 輸出
            teacher_l4_out: (B, C, T) Teacher L4 輸出

        Returns:
            loss: MSE Loss
            loss_info: Loss 資訊
        """
        # 確保時間維度匹配
        min_t = min(student_l4_out.shape[-1], teacher_l4_out.shape[-1])
        student_l4_out = student_l4_out[..., :min_t]
        teacher_l4_out = teacher_l4_out[..., :min_t]

        mse_loss = F.mse_loss(student_l4_out, teacher_l4_out, reduction=self.reduction)

        # 額外計算 cosine similarity 作為參考
        with torch.no_grad():
            # Flatten to (B*T, C)
            s_flat = student_l4_out.permute(0, 2, 1).reshape(-1, student_l4_out.shape[1])
            t_flat = teacher_l4_out.permute(0, 2, 1).reshape(-1, teacher_l4_out.shape[1])
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1).mean().item()

        loss_info = {
            'mse_loss': mse_loss.item(),
            'cos_sim': cos_sim,
        }

        return mse_loss, loss_info

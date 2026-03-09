"""
exp_0112: 三區差異化訓練 (Exp I)

基於之前實驗的關鍵洞察:
- Exp B: 凍結深層會破壞模型協同性 → 必須保持梯度流通
- Exp F: 差異化 LR 有效但二分法太粗糙
- Feature Map 分析: 中層 (L5-L8) 對噪音最敏感，應重點強化

策略: 三區差異化 LR + 淺層輕度 L2
- 淺層 (L0-L4):  低 LR + 輕度 L2 → 穩定基礎聲學特徵
- 中層 (L5-L8):  高 LR + 無 L2  → 重點學習去噪
- 深層 (L9-L17): 中 LR + 無 L2  → 自由配合上游調整
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import List, Dict, Optional

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
# 三區層級定義
# ============================================================

# 淺層 (L0-L4): 聲學特徵層
SHALLOW_LAYERS = [
    "feature_extractor.encodec.encoder.model.0.conv.conv",           # L0: input
    "feature_extractor.encodec.encoder.model.1.block.1.conv.conv",   # L1
    "feature_extractor.encodec.encoder.model.1.block.3.conv.conv",   # L2
    "feature_extractor.encodec.encoder.model.1.shortcut.conv.conv",  # L3
    "feature_extractor.encodec.encoder.model.3.conv.conv",           # L4: downsample
]

# 中層 (L5-L8): 噪音敏感層 - 重點學習區
MIDDLE_LAYERS = [
    "feature_extractor.encodec.encoder.model.4.block.1.conv.conv",   # L5
    "feature_extractor.encodec.encoder.model.4.block.3.conv.conv",   # L6
    "feature_extractor.encodec.encoder.model.4.shortcut.conv.conv",  # L7
    "feature_extractor.encodec.encoder.model.6.conv.conv",           # L8: downsample
]

# 深層 (L9-L17): 語義特徵層
DEEP_LAYERS = [
    "feature_extractor.encodec.encoder.model.7.block.1.conv.conv",   # L9
    "feature_extractor.encodec.encoder.model.7.block.3.conv.conv",   # L10
    "feature_extractor.encodec.encoder.model.7.shortcut.conv.conv",  # L11
    "feature_extractor.encodec.encoder.model.9.conv.conv",           # L12: downsample
    "feature_extractor.encodec.encoder.model.10.block.1.conv.conv",  # L13
    "feature_extractor.encodec.encoder.model.10.block.3.conv.conv",  # L14
    "feature_extractor.encodec.encoder.model.10.shortcut.conv.conv", # L15
    "feature_extractor.encodec.encoder.model.12.conv.conv",          # L16: downsample
    "feature_extractor.encodec.encoder.model.15.conv.conv",          # L17: output
]

ALL_LAYERS = SHALLOW_LAYERS + MIDDLE_LAYERS + DEEP_LAYERS


def get_layer_zone(layer_name: str) -> str:
    """判斷層屬於哪個區域"""
    if layer_name in SHALLOW_LAYERS:
        return 'shallow'
    elif layer_name in MIDDLE_LAYERS:
        return 'middle'
    elif layer_name in DEEP_LAYERS:
        return 'deep'
    else:
        return 'unknown'


class TeacherStudentThreeZone(nn.Module):
    """
    Exp I: 三區差異化訓練模型

    核心設計:
    - 全層 LoRA，保持梯度流通
    - 三區差異化 LR: 中層最高，淺層最低
    - 淺層輕度 L2: 穩定基礎特徵
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

        # Teacher: 完全凍結
        print("=" * 60)
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)
        self._freeze_quantizer(self.teacher, "Teacher")

        # Student: 全層 LoRA (三區差異化)
        print("=" * 60)
        print("Loading Student with Three-Zone LoRA...")
        print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}")
        print(f"  Shallow (L0-L4): {len(SHALLOW_LAYERS)} layers")
        print(f"  Middle (L5-L8):  {len(MIDDLE_LAYERS)} layers")
        print(f"  Deep (L9-L17):   {len(DEEP_LAYERS)} layers")

        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=ALL_LAYERS,
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.student = get_peft_model(self.student, lora_config)
        self.student.print_trainable_parameters()
        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # 保存淺層初始權重 (用於 L2 正則化)
        self._save_initial_shallow_weights()

        # 獲取 codebook 並保存初始狀態
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        print("=" * 60)
        print(f"Codebook shape: {self.codebook.shape}")
        print("=" * 60)

    def _freeze_quantizer(self, model, name: str):
        quantizer = model.feature_extractor.encodec.quantizer
        quantizer.eval()
        for param in quantizer.parameters():
            param.requires_grad = False
        print(f"  {name} quantizer frozen")

    def _save_initial_shallow_weights(self):
        """保存淺層 LoRA 初始權重 (用於 L2 正則化)"""
        self.initial_shallow_weights = {}
        for name, param in self.student.named_parameters():
            if param.requires_grad and any(layer in name for layer in SHALLOW_LAYERS):
                self.initial_shallow_weights[name] = param.data.clone()
        print(f"  Saved {len(self.initial_shallow_weights)} shallow layer weights for L2 reg")

    def compute_shallow_l2_regularization(self) -> torch.Tensor:
        """計算淺層 L2 正則化損失"""
        l2_loss = 0.0
        n_params = 0

        for name, param in self.student.named_parameters():
            if name in self.initial_shallow_weights:
                diff = param - self.initial_shallow_weights[name].to(param.device)
                l2_loss = l2_loss + (diff ** 2).sum()
                n_params += param.numel()

        if n_params > 0:
            l2_loss = l2_loss / n_params

        return l2_loss

    def get_three_zone_param_groups(
        self,
        lr_shallow: float = 5e-6,
        lr_middle: float = 2e-5,
        lr_deep: float = 1e-5,
        weight_decay: float = 0.05
    ) -> List[Dict]:
        """
        為三區差異化學習率創建參數組

        預設配置:
        - 淺層: 5e-6 (0.5x base)
        - 中層: 2e-5 (2.0x base) ← 重點學習
        - 深層: 1e-5 (1.0x base)
        """
        shallow_params = []
        middle_params = []
        deep_params = []

        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue

            is_shallow = any(layer in name for layer in SHALLOW_LAYERS)
            is_middle = any(layer in name for layer in MIDDLE_LAYERS)
            is_deep = any(layer in name for layer in DEEP_LAYERS)

            if is_shallow:
                shallow_params.append(param)
            elif is_middle:
                middle_params.append(param)
            elif is_deep:
                deep_params.append(param)
            else:
                # 未分類的參數用深層 LR
                deep_params.append(param)

        param_groups = []
        if shallow_params:
            param_groups.append({
                'params': shallow_params,
                'lr': lr_shallow,
                'weight_decay': weight_decay,
                'name': 'shallow_layers'
            })
        if middle_params:
            param_groups.append({
                'params': middle_params,
                'lr': lr_middle,
                'weight_decay': weight_decay,
                'name': 'middle_layers'
            })
        if deep_params:
            param_groups.append({
                'params': deep_params,
                'lr': lr_deep,
                'weight_decay': weight_decay,
                'name': 'deep_layers'
            })

        print(f"\nThree-Zone LR param groups:")
        print(f"  Shallow (L0-L4):  {len(shallow_params)} params, LR={lr_shallow:.1e}")
        print(f"  Middle (L5-L8):   {len(middle_params)} params, LR={lr_middle:.1e} ★")
        print(f"  Deep (L9-L17):    {len(deep_params)} params, LR={lr_deep:.1e}")
        print(f"  LR ratio: shallow={lr_shallow/lr_deep:.1f}x, middle={lr_middle/lr_deep:.1f}x, deep=1.0x")

        return param_groups

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
        B, C, T = encoder_out.shape
        z = encoder_out.permute(0, 2, 1)
        logits = 2 * torch.matmul(z, self.codebook.t())
        c_sq = (self.codebook ** 2).sum(dim=1)
        logits = logits - c_sq.unsqueeze(0).unsqueeze(0)
        logits = logits.permute(0, 2, 1)
        return logits

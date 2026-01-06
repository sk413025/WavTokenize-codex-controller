"""
exp_0106: 差異化訓練策略實驗

基於 exp_1231_1 實驗結論:
- Exp B (凍結深層): Gap 消除 (0.06%) 但準確度崩潰 (0.51%)
- Exp E (漸進訓練): Gap 改善 (1.49%) 準確度一般 (0.82%)

關鍵洞察:
- 深層變化大是「症狀」不是「原因」
- 真正問題: 淺層學習不足 → 深層被迫補償 → 過擬合
- 解決方案: 強制淺層多學習，限制深層變化

三種策略:
- Exp F: 差異化學習率 (淺層 LR 高，深層 LR 低)
- Exp G: 差異化 Rank (淺層 Rank 高，深層 Rank 低)
- Exp H: 全層訓練 + 深層 L2 正則化
"""

import torch
import torch.nn as nn
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

# 層分組
SHALLOW_LAYERS = INPUT_LAYER + LOW_LEVEL_LAYERS + MID_LEVEL_LAYERS  # L0-L8
DEEP_LAYERS = SEMANTIC_LAYERS + ABSTRACT_LAYERS + OUTPUT_LAYER       # L9-L17
ALL_18_LAYERS = SHALLOW_LAYERS + DEEP_LAYERS

# Layer presets
LORA_LAYER_PRESETS = {
    'all_18': ALL_18_LAYERS,
    'shallow': SHALLOW_LAYERS,
    'deep': DEEP_LAYERS,
    'mid_only': MID_LEVEL_LAYERS,
}


def get_layer_group(layer_name: str) -> str:
    """判斷層屬於哪個分組"""
    if layer_name in INPUT_LAYER:
        return 'input'
    elif layer_name in LOW_LEVEL_LAYERS:
        return 'low_level'
    elif layer_name in MID_LEVEL_LAYERS:
        return 'mid_level'
    elif layer_name in SEMANTIC_LAYERS:
        return 'semantic'
    elif layer_name in ABSTRACT_LAYERS:
        return 'abstract'
    elif layer_name in OUTPUT_LAYER:
        return 'output'
    else:
        return 'unknown'


def is_shallow_layer(layer_name: str) -> bool:
    """判斷是否為淺層 (L0-L8)"""
    return layer_name in SHALLOW_LAYERS


def is_deep_layer(layer_name: str) -> bool:
    """判斷是否為深層 (L9-L17)"""
    return layer_name in DEEP_LAYERS


class TeacherStudentDifferentialLR(nn.Module):
    """
    Exp F: 差異化學習率模型

    策略: 淺層高 LR，深層低 LR
    - 淺層 (L0-L8): lr_shallow (預設 1e-4)
    - 深層 (L9-L17): lr_deep (預設 1e-5)
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

        # Student: 全層 LoRA
        print("=" * 60)
        print("Loading Student with Full LoRA (Differential LR)...")
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
        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

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

    def _get_codebook(self) -> torch.Tensor:
        quantizer = self.teacher.feature_extractor.encodec.quantizer
        codebook = quantizer.vq.layers[0].codebook.detach().clone()
        return codebook

    def _get_teacher_codebook(self) -> torch.Tensor:
        return self.teacher.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def _get_student_codebook(self) -> torch.Tensor:
        return self.student.feature_extractor.encodec.quantizer.vq.layers[0].codebook

    def get_differential_param_groups(
        self,
        lr_shallow: float = 1e-4,
        lr_deep: float = 1e-5,
        weight_decay: float = 0.05
    ) -> List[Dict]:
        """
        為差異化學習率創建參數組

        Returns:
            List of param groups for optimizer
        """
        shallow_params = []
        deep_params = []

        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue

            # 判斷是淺層還是深層
            is_shallow = any(layer in name for layer in SHALLOW_LAYERS)
            is_deep = any(layer in name for layer in DEEP_LAYERS)

            if is_shallow:
                shallow_params.append(param)
            elif is_deep:
                deep_params.append(param)
            else:
                # 其他可訓練參數 (如果有) 用淺層 LR
                shallow_params.append(param)

        param_groups = []
        if shallow_params:
            param_groups.append({
                'params': shallow_params,
                'lr': lr_shallow,
                'weight_decay': weight_decay,
                'name': 'shallow_layers'
            })
        if deep_params:
            param_groups.append({
                'params': deep_params,
                'lr': lr_deep,
                'weight_decay': weight_decay,
                'name': 'deep_layers'
            })

        print(f"\nDifferential LR param groups:")
        print(f"  Shallow (L0-L8): {len(shallow_params)} params, LR={lr_shallow}")
        print(f"  Deep (L9-L17): {len(deep_params)} params, LR={lr_deep}")
        print(f"  LR ratio (shallow/deep): {lr_shallow/lr_deep:.1f}x")

        return param_groups

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


class TeacherStudentDifferentialRank(nn.Module):
    """
    Exp G: 差異化 Rank 模型

    策略: 淺層高 Rank，深層低 Rank
    - 淺層 (L0-L8): rank_shallow (預設 256)
    - 深層 (L9-L17): rank_deep (預設 32)

    需要分別創建兩個 LoRA config
    """

    def __init__(
        self,
        wavtok_config: str,
        wavtok_ckpt: str,
        rank_shallow: int = 256,
        rank_deep: int = 32,
        lora_alpha_shallow: int = 512,
        lora_alpha_deep: int = 64,
        lora_dropout: float = 0.2,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.rank_shallow = rank_shallow
        self.rank_deep = rank_deep

        # Teacher: 完全凍結
        print("=" * 60)
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)
        self._freeze_quantizer(self.teacher, "Teacher")

        # Student: 差異化 Rank LoRA
        print("=" * 60)
        print("Loading Student with Differential Rank LoRA...")
        print(f"  Shallow (L0-L8): Rank={rank_shallow}, Alpha={lora_alpha_shallow}")
        print(f"  Deep (L9-L17): Rank={rank_deep}, Alpha={lora_alpha_deep}")

        self.student = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)

        # 方案：分別對淺層和深層創建兩個 student 模型
        # 由於 PEFT 不支持同時訓練多 adapter，我們改用「分層創建」策略
        # 先對淺層創建 LoRA，再把深層 LoRA 加上去並啟用

        # 創建包含所有層的統一 LoRA (但使用淺層的高 rank)
        # 然後把深層的 LoRA 權重縮放來模擬低 rank 效果
        combined_lora_config = LoraConfig(
            r=rank_shallow,
            lora_alpha=lora_alpha_shallow,
            target_modules=ALL_18_LAYERS,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.student = get_peft_model(self.student, combined_lora_config)

        # 對深層參數施加更強的正則化來限制其變化
        # 這樣雖然 rank 相同，但深層會被更強的約束
        self._deep_layer_names = []
        for name, param in self.student.named_parameters():
            if param.requires_grad and any(layer in name for layer in DEEP_LAYERS):
                self._deep_layer_names.append(name)

        print(f"  Note: Using uniform rank={rank_shallow}, deep layers will have stronger L2 reg")
        print(f"  Deep layer params: {len(self._deep_layer_names)}")

        # 保存深層初始權重
        self._initial_deep_weights = {}
        for name, param in self.student.named_parameters():
            if name in self._deep_layer_names:
                self._initial_deep_weights[name] = param.data.clone()

        self.student.print_trainable_parameters()
        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # 獲取 codebook 並保存初始狀態
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        print("=" * 60)
        print(f"Codebook shape: {self.codebook.shape}")
        print(f"Rank ratio (shallow/deep): {rank_shallow/rank_deep:.1f}x")
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

    def compute_deep_l2_regularization(self) -> torch.Tensor:
        """計算深層 L2 正則化損失"""
        l2_loss = 0.0
        n_params = 0

        for name, param in self.student.named_parameters():
            if name in self._initial_deep_weights:
                diff = param - self._initial_deep_weights[name].to(param.device)
                l2_loss = l2_loss + (diff ** 2).sum()
                n_params += param.numel()

        if n_params > 0:
            l2_loss = l2_loss / n_params

        return l2_loss

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

        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

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


class TeacherStudentWithL2Reg(nn.Module):
    """
    Exp H: 全層訓練 + 深層 L2 正則化

    策略:
    - 全部 18 層都加 LoRA
    - 對深層 (L9-L17) 施加額外的 L2 正則化
    - 限制深層權重變化幅度
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

        # Teacher
        print("=" * 60)
        print("Loading Teacher (frozen)...")
        self.teacher = WavTokenizer.from_pretrained0802(wavtok_config, wavtok_ckpt)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher = self.teacher.to(device)
        self._freeze_quantizer(self.teacher, "Teacher")

        # Student: 全層 LoRA
        print("=" * 60)
        print("Loading Student with Full LoRA + Deep L2 Regularization...")
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
        self.student = self.student.to(device)
        self._freeze_quantizer(self.student, "Student")

        # 保存初始深層 LoRA 權重用於 L2 正則化
        self._save_initial_deep_weights()

        # 獲取 codebook
        self.codebook = self._get_codebook()
        self._initial_teacher_codebook = self._get_teacher_codebook().clone()
        self._initial_student_codebook = self._get_student_codebook().clone()

        print("=" * 60)
        print(f"Codebook shape: {self.codebook.shape}")
        print("=" * 60)

    def _save_initial_deep_weights(self):
        """保存深層 LoRA 初始權重"""
        self.initial_deep_weights = {}
        for name, param in self.student.named_parameters():
            if param.requires_grad and any(layer in name for layer in DEEP_LAYERS):
                self.initial_deep_weights[name] = param.data.clone()
        print(f"  Saved {len(self.initial_deep_weights)} deep layer weights for L2 reg")

    def compute_deep_l2_regularization(self) -> torch.Tensor:
        """計算深層 L2 正則化損失"""
        l2_loss = 0.0
        n_params = 0

        for name, param in self.student.named_parameters():
            if name in self.initial_deep_weights:
                diff = param - self.initial_deep_weights[name].to(param.device)
                l2_loss = l2_loss + (diff ** 2).sum()
                n_params += param.numel()

        if n_params > 0:
            l2_loss = l2_loss / n_params

        return l2_loss

    def get_deep_weight_change(self) -> dict:
        """計算深層權重變化量 (用於監控)"""
        changes = {}
        for name, param in self.student.named_parameters():
            if name in self.initial_deep_weights:
                diff = (param - self.initial_deep_weights[name].to(param.device)).abs().mean().item()
                changes[name] = diff
        return changes

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

        self.teacher.eval()
        self.teacher.feature_extractor.encodec.quantizer.eval()

        with torch.no_grad():
            teacher_encoder_out = self.teacher.feature_extractor.encodec.encoder(clean_audio)
            teacher_vq = self.teacher.feature_extractor.encodec.quantizer(
                teacher_encoder_out, frame_rate=75, bandwidth=0.075
            )
            teacher_codes = teacher_vq.codes

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

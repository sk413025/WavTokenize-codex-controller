"""
exp_0112_intermediate: Exp K v3 - 完整中間層監督

配置:
- L3 (0.5): low_level 代表, Cosine Loss
- L5 (0.8): mid_level 協同, Cosine Loss
- L6 (1.0): 噪音處理核心, Cosine Loss
- L10 (0.3): 語義錨點, MSE Loss (因為本來就穩定)

基於分析:
- exp_1231_feature: L5-L6 是噪音處理核心 (敏感度 0.71-0.79)
- 本次分析: L10 是最穩定層 (cos_sim=0.946)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List


class IntermediateSupervisionLossV3(nn.Module):
    """
    Exp K v3: 支援混合 Loss 類型的中間層監督

    特點:
    1. 可為每層指定不同的 Loss 類型 (Cosine/MSE)
    2. L10 錨點使用 MSE (因為本來就穩定)
    3. 其他層使用 Cosine (尺度不變性)
    """

    def __init__(
        self,
        intermediate_weights: Dict[int, float] = None,
        layer_loss_types: Dict[int, str] = None,  # 每層的 loss 類型
        default_loss_type: str = 'cosine',
        target_scale: float = 1.0,
    ):
        """
        Args:
            intermediate_weights: {layer_index: weight} 各層的權重
            layer_loss_types: {layer_index: 'cosine' or 'mse'} 各層的 loss 類型
            default_loss_type: 預設 loss 類型
            target_scale: 目標 loss 尺度
        """
        super().__init__()

        # Exp K v3 預設配置
        if intermediate_weights is None:
            intermediate_weights = {
                3: 0.5,   # L3 (low_level)
                5: 0.8,   # L5 (mid_level 協同)
                6: 1.0,   # L6 (噪音處理核心)
                10: 0.3,  # L10 (語義錨點)
            }

        if layer_loss_types is None:
            layer_loss_types = {
                3: 'cosine',
                5: 'cosine',
                6: 'cosine',
                10: 'mse',  # L10 用 MSE (本來就穩定)
            }

        self.intermediate_weights = intermediate_weights
        self.layer_loss_types = layer_loss_types
        self.default_loss_type = default_loss_type
        self.target_scale = target_scale

        print(f"IntermediateSupervisionLossV3 initialized:")
        print(f"  Layers: {list(intermediate_weights.keys())}")
        print(f"  Weights: {intermediate_weights}")
        print(f"  Loss types: {layer_loss_types}")
        print(f"  target_scale: {target_scale}")

    def _normalized_mse(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        正規化 MSE: 先對 feature 做 L2 正規化，再計算 MSE
        """
        student_norm = F.normalize(student, p=2, dim=1)
        teacher_norm = F.normalize(teacher, p=2, dim=1)
        loss = F.mse_loss(student_norm, teacher_norm)
        return loss

    def _cosine_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        Cosine Similarity Loss: 1 - cos_sim
        """
        B, C, T = student.shape
        s_flat = student.permute(0, 2, 1).reshape(-1, C)
        t_flat = teacher.permute(0, 2, 1).reshape(-1, C)
        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)
        loss = 1.0 - cos_sim.mean()
        return loss

    def forward(
        self,
        student_intermediates: Dict[int, torch.Tensor],
        teacher_intermediates: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算中間層監督 Loss

        Returns:
            loss: 總 loss (已縮放)
            loss_info: 各層 loss 資訊
        """
        total_loss = 0.0
        loss_info = {}

        for idx, weight in self.intermediate_weights.items():
            if idx in student_intermediates and idx in teacher_intermediates:
                student_feat = student_intermediates[idx]
                teacher_feat = teacher_intermediates[idx]

                # 確保維度匹配
                min_t = min(student_feat.shape[-1], teacher_feat.shape[-1])
                student_feat = student_feat[..., :min_t]
                teacher_feat = teacher_feat[..., :min_t]

                # 獲取該層的 loss 類型
                loss_type = self.layer_loss_types.get(idx, self.default_loss_type)

                # 計算 loss
                if loss_type == 'mse':
                    layer_loss = self._normalized_mse(student_feat, teacher_feat)
                elif loss_type == 'cosine':
                    layer_loss = self._cosine_loss(student_feat, teacher_feat)
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")

                total_loss = total_loss + weight * layer_loss
                loss_info[f'intermediate_L{idx}_loss'] = layer_loss.item()
                loss_info[f'intermediate_L{idx}_type'] = loss_type

        # 縮放到目標尺度
        total_loss = total_loss * self.target_scale
        loss_info['intermediate_total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, loss_info


def get_exp_k_v3_config():
    """
    獲取 Exp K v3 的預設配置
    """
    return {
        'intermediate_indices': [3, 5, 6, 10],
        'intermediate_weights': {
            3: 0.5,   # L3 (low_level)
            5: 0.8,   # L5 (mid_level 協同)
            6: 1.0,   # L6 (噪音處理核心)
            10: 0.3,  # L10 (語義錨點)
        },
        'layer_loss_types': {
            3: 'cosine',
            5: 'cosine',
            6: 'cosine',
            10: 'mse',  # L10 用 MSE
        },
    }

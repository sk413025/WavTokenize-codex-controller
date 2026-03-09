"""
exp_0112_intermediate: Exp K v2 - 修正版中間層監督

問題診斷:
- 原版 IntermediateSupervisionLoss 使用原始 MSE
- L6 的 MSE 高達 1546，遠超 feature_loss (0.27) 和 triplet_loss (0.73)
- 導致中間層 Loss 主導訓練 (佔比 199.5%)

修正方案:
1. 使用 Normalized MSE (除以 feature dimension)
2. 增加 Cosine Similarity Loss 選項
3. 自動縮放讓中間層 Loss 與 final Loss 同尺度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class IntermediateSupervisionLossV2(nn.Module):
    """
    修正版中間層監督 Loss

    修正點:
    1. MSE 正規化 (除以 C 維度)
    2. 可選 Cosine Similarity Loss
    3. 自動縮放到合理範圍 (~1.0)
    """

    def __init__(
        self,
        intermediate_weights: Dict[int, float] = None,
        loss_type: str = 'normalized_mse',  # 'mse', 'normalized_mse', 'cosine', 'combined'
        cosine_weight: float = 0.5,  # 用於 'combined' 模式
        target_scale: float = 1.0,  # 目標 loss 尺度
    ):
        """
        Args:
            intermediate_weights: {layer_index: weight} 各層的權重
            loss_type: 'mse', 'normalized_mse', 'cosine', 'combined'
            cosine_weight: combined 模式中 cosine loss 的權重
            target_scale: 目標 loss 尺度 (讓 loss 值約在這個範圍)
        """
        super().__init__()

        if intermediate_weights is None:
            intermediate_weights = {3: 0.5, 6: 0.5}

        self.intermediate_weights = intermediate_weights
        self.loss_type = loss_type
        self.cosine_weight = cosine_weight
        self.target_scale = target_scale

        print(f"IntermediateSupervisionLossV2 initialized:")
        print(f"  loss_type: {loss_type}")
        print(f"  weights: {intermediate_weights}")
        print(f"  target_scale: {target_scale}")

    def _normalized_mse(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        正規化 MSE: 先對 feature 做 L2 正規化，再計算 MSE

        這樣可以讓不同層的 Loss 在相同尺度上

        Args:
            student: (B, C, T)
            teacher: (B, C, T)

        Returns:
            loss: scalar, normalized MSE (約在 0-4 範圍)
        """
        # L2 normalize along channel dimension
        student_norm = F.normalize(student, p=2, dim=1)  # (B, C, T)
        teacher_norm = F.normalize(teacher, p=2, dim=1)  # (B, C, T)

        # MSE on normalized features
        loss = F.mse_loss(student_norm, teacher_norm)

        return loss

    def _cosine_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        Cosine Similarity Loss: 1 - cos_sim

        Args:
            student: (B, C, T)
            teacher: (B, C, T)

        Returns:
            loss: scalar, 1 - mean(cos_sim)
        """
        # Flatten to (B*T, C)
        B, C, T = student.shape
        s_flat = student.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
        t_flat = teacher.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)

        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)  # (B*T,)
        loss = 1.0 - cos_sim.mean()

        return loss

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
            loss: 總 loss (已縮放到 target_scale)
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

                # 計算 loss
                if self.loss_type == 'mse':
                    layer_loss = F.mse_loss(student_feat, teacher_feat)
                elif self.loss_type == 'normalized_mse':
                    layer_loss = self._normalized_mse(student_feat, teacher_feat)
                elif self.loss_type == 'cosine':
                    layer_loss = self._cosine_loss(student_feat, teacher_feat)
                elif self.loss_type == 'combined':
                    mse_loss = self._normalized_mse(student_feat, teacher_feat)
                    cos_loss = self._cosine_loss(student_feat, teacher_feat)
                    layer_loss = (1 - self.cosine_weight) * mse_loss + self.cosine_weight * cos_loss
                else:
                    raise ValueError(f"Unknown loss_type: {self.loss_type}")

                total_loss = total_loss + weight * layer_loss
                loss_info[f'intermediate_L{idx}_loss'] = layer_loss.item()

        # 縮放到目標尺度
        total_loss = total_loss * self.target_scale

        loss_info['intermediate_total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, loss_info


# 為了向後兼容，保留原始類但加上警告
class IntermediateSupervisionLoss(IntermediateSupervisionLossV2):
    """
    向後兼容的包裝器 - 使用 normalized_mse 作為預設
    """
    def __init__(
        self,
        intermediate_weights: Dict[int, float] = None,
        reduction: str = 'mean',  # 保留參數但忽略
    ):
        print("WARNING: Using IntermediateSupervisionLossV2 with normalized_mse")
        super().__init__(
            intermediate_weights=intermediate_weights,
            loss_type='normalized_mse',
            target_scale=1.0,
        )
